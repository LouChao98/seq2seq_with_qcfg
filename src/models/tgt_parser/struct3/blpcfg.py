import torch

from ._utils import checkpoint
from .pcfgs import PCFG_base


class FastBLPCFG(PCFG_base):
    # head:                 b, N, NT, r     N,NT -> r
    # inherent:             b, NT_T, r.     r -> NT_T       get word from parent
    # noninherent_symbol:   b, NT_T, r, 2.  r -> 2 * NT_T
    # noninherent_word:     b, N, r.        r -> N
    # root:                 b, N, NT.       S -> NT,N

    def __init__(self):
        super(FastBLPCFG, self).__init__()

    def loss(self, rules, lens):
        return self._inside(rules, lens)

    @torch.enable_grad()
    def decode(self, rules, lens, mbr=True, viterbi=False, eval_dep=False):
        noninherent_word = rules["noninherent_word"]
        root = rules["root"]
        noninherent_symbol = rules["noninherent_symbol"]
        head = rules["head"]
        inherent = rules["inherent"]
        b, n = root.shape[:2]
        arc_indicator = inherent.new_zeros(b, n, n).requires_grad_(True)
        span_indicator = inherent.new_zeros(b, n + 1, n + 1).requires_grad_(True)
        logZ = Inside.apply(
            noninherent_word, noninherent_symbol, inherent, root, head, span_indicator, arc_indicator, lens
        )
        logZ.sum().backward()

        # to avoid some trivial corner cases.
        if n >= 3:
            marginals = span_indicator.grad
            prediction = self._cky_zero_order(marginals.detach(), lens)
        else:
            prediction = [[] for _ in range(b)]

        # TODO:
        if eval_dep:
            pass

        return {"partition": logZ, "prediction": prediction}

    @torch.enable_grad()
    def _inside(self, rules, lens, **kwargs):
        root = rules["root"]
        head = rules["head"]
        noninherent_word = rules["noninherent_word"]
        noninherent_symbol = rules["noninherent_symbol"]
        inherent = rules["inherent"]
        b, n = root.shape[:2]
        arc_indicator = inherent.new_zeros(b, n, n)
        span_indicator = inherent.new_zeros(b, n + 1, n + 1)
        logZ = Inside.apply(
            noninherent_word, noninherent_symbol, inherent, root, head, arc_indicator, span_indicator, lens
        )
        return {"partition": logZ}


class Inside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, noninherent_word, noninherent_symbol, inherent, root, head, span_indicator, arc_indicator, lens):
        H = root.shape[1]
        NT = root.shape[-1]
        S = noninherent_symbol.shape[1]
        r = inherent.shape[-1]
        B = root.shape[0]
        N = H + 1
        LEFT = 0
        RIGHT = 1
        nt_slice = slice(0, NT)
        t_slice = slice(NT, S)

        # s = head.new_zeros(B, N, N, H, NT).fill_(0)
        s_normalizer = head.new_full((B, N, N, H), -1e9)
        s_noninherent = head.new_zeros(B, N, N, r, 2)
        s = head.new_full((B, N, N, H, NT), -1e9)
        s_inherent = head.new_zeros(B, N, N, H, r)
        noninherent_nt = noninherent_symbol[:, nt_slice, :].contiguous()
        inherent_nt = inherent[:, nt_slice].contiguous()

        arange_H = torch.arange(H)
        symbol_word = (noninherent_symbol[:, None, t_slice].sum(1) + 1e-9).log() + noninherent_word[..., None]
        s_noninherent[:, arange_H, arange_H + 1, :] = symbol_word
        s_inherent[:, arange_H, arange_H + 1, arange_H] = (
            inherent[:, t_slice, :].sum(-2).unsqueeze(1).expand(-1, H, -1)
        )
        s_normalizer[:, arange_H, arange_H + 1, arange_H] = 0

        # calculation of s_{i,j}^{A,p} [Term A1 (left) + Term A2 (right)] in Equation (3) of the paper.

        def merge(left, right, closed_left, closed_right, head_rule):
            headed = Operation1.apply(left, right, closed_left, closed_right)
            headed_normalizer = headed.max(-1)[0]
            headed = (headed - headed_normalizer.unsqueeze(-1)).exp()
            new_headed = torch.einsum("bnhr,bnhmr->bnhm", headed, head_rule)
            return headed_normalizer, new_headed

        def contract_qC(s, rule_word, rule_symbol, normalizer):
            """
            contract the head word and nt of the noninherent span.
            :param s:  inside score of shape (b, n, h, NT)
            :param rule_symbol: of shape (b, n, r, D)  [log P(H->C,D)] noninherent nt
            :param rule_word: of shape (b, n, h, r)  [log P(H->beta)] beta
            :return: (b, n, r)
            """
            # b, n, nt, r, 2
            tmp = torch.einsum("bnhm, bmrd->bnhrd", s, rule_symbol)
            tmp = (tmp.clamp(min=1e-9)).log() + normalizer[..., None, None]
            return (tmp + rule_word.unsqueeze(-1)).logsumexp(2)

        # Term D-1-1 in the paper
        @checkpoint
        def contract_B(s, rule):
            """
            :param s: inside score of shape (b, n, h, NT)
            :param rule: shape (b, NT, r),  log p(H -> B) inherent nt.
            :return: shape (b, n, h, r)
            """
            return torch.einsum("bnhm, bmr -> bnhr", s, rule)

        for w in range(2, N):
            n = N - w
            # Equation (8).  Term D1-1 \times Term D1-2
            # (b, n, w, H, r)
            # right = stripe_with_headword(s_inherent, n, w - 1, (1, w), 0)
            # (b, n, w, r)
            closed_left = stripe(s_noninherent[..., LEFT], n, w - 1, (0, 1))
            closed_right = stripe(s_noninherent[..., RIGHT], n, w - 1, (1, w), 0)
            left_normalizer = stripe_with_headword(s_normalizer, n, w - 1, (0, 1))
            right_normalizer = stripe_with_headword(s_normalizer, n, w - 1, (1, w), 0)
            # (b, n, H, A)
            headed_normalizer, headed = merge(
                (stripe_with_headword(s_inherent, n, w - 1, (0, 1)).clamp(min=1e-9))
                .log_()
                .add_(left_normalizer.unsqueeze(-1)),
                (stripe_with_headword(s_inherent, n, w - 1, (1, w), 0).clamp(min=1e-9))
                .log_()
                .add_(right_normalizer.unsqueeze(-1)),
                closed_left,
                closed_right,
                stripe_grammar_rules(head, n, w),
            )

            diagonal_copy_with_headword(s, headed, w)
            diagonal_copy_with_headword(s_normalizer, headed_normalizer, w)

            if w < N - 1:
                # calculating Term D1-2 as described in the paper.
                headed_closed = contract_qC(
                    headed, stripe_grammar_rules(noninherent_word, n, w), noninherent_nt, headed_normalizer
                )
                # calculating Term D1-1 as described in the paper.
                headed = contract_B(headed, inherent_nt)
                # caching them.
                diagonal_copy_(s_noninherent, headed_closed, w)
                diagonal_copy_with_headword(s_inherent, headed, w)

        tmp = (
            (s[torch.arange(B), 0, lens].clamp(min=1e-9))
            .log_()
            .add_(headed_normalizer.squeeze(1).unsqueeze(-1))
            .add_(root)
        )
        logZ = (tmp).logsumexp([-1, -2])

        grad_noninherent_word = logZ.new_zeros(*noninherent_word.shape)
        grad_noninherent_symbol = logZ.new_zeros(*noninherent_symbol.shape)
        grad_inherent = logZ.new_zeros(*inherent.shape)
        grad_head = logZ.new_zeros(*head.shape)

        grad_s = logZ.new_zeros(*s.shape)
        grad_span_headword_left = logZ.new_zeros(B, N, N, H, r)
        grad_span_headword_right = logZ.new_zeros(B, N, N, H, r)

        grad_s_noninherent_l = logZ.new_zeros(B, N, N, r)
        grad_s_noninherent_r = logZ.new_zeros(B, N, N, r)
        grad_s_inherent = logZ.new_zeros(B, N, N, H, r)

        # marginals.
        grad_arc = logZ.new_zeros(B, H, H)
        grad_span = logZ.new_zeros(B, N, N)

        grad_root = (tmp - logZ[..., None, None]).exp_()
        grad_s[torch.arange(B), 0, lens] = (1 / (s[torch.arange(B), 0, lens].clamp(min=1e-9))) * grad_root

        # aaa = ((1 / (headed + 1e-9)) * grad_root).sum(1)

        for w in range(N - 1, 1, -1):
            n = N - w
            if w < N - 1:
                # parent gradient..
                tmp_grd_s_inherent = diagonal_with_headword(grad_s_inherent, w)
                tmp_grd_s_noninherent = torch.cat(
                    [
                        diagonal(grad_s_noninherent_l, w).unsqueeze(-1),
                        diagonal(grad_s_noninherent_r, w).unsqueeze(-1),
                    ],
                    dim=-1,
                )

                # Backward Contract_B
                headed = diagonal_with_headword(s, w)
                grad_inherent[:, nt_slice] += torch.einsum("bnhr, bnhm -> bmr", tmp_grd_s_inherent, headed)
                tmp_grd_headed = torch.einsum("bnhr, bmr->bnhm", tmp_grd_s_inherent, inherent_nt)

                # compute arc marginal.
                # Forward Contract Q_C
                headed_normalizer = diagonal_with_headword(s_normalizer, w)
                rule = stripe_grammar_rules(noninherent_word, n, w)
                tmp_bnhrd_1 = torch.einsum("bnhm, bmrd->bnhrd", headed, noninherent_nt)
                tmp_bnhrd_2 = (tmp_bnhrd_1.clamp(min=1e-9)).log_().add_(headed_normalizer[..., None, None])
                tmp = tmp_bnhrd_2 + rule.unsqueeze(-1)
                final = (tmp).logsumexp(2)
                percent = (tmp.sub_(final.unsqueeze(2))).exp_()

                # Backward Contract Q_C
                grd_tmp = percent * tmp_grd_s_noninherent[:, :, None, ...]
                grd_rule_word = grd_tmp.sum(-1)
                for i in range(n):
                    grad_noninherent_word[:, i : i + w] += grd_rule_word[:, i]

                tmptmp = (1 / tmp_bnhrd_1.clamp(min=1e-9)) * grd_tmp
                grad_noninherent_symbol[:, nt_slice] += torch.einsum("bnhrd, bnhm -> bmrd", tmptmp, headed)
                tmp_grd_headed += torch.einsum("bnhrd, bmrd -> bnhm", tmptmp, noninherent_nt)
                diagonal_with_headword_add_(grad_s, tmp_grd_headed, w)

                tmp_grd = torch.cat(
                    [
                        diagonal(grad_span_headword_left, w).unsqueeze(-1),
                        diagonal(grad_span_headword_right, w).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                arc_grd = torch.einsum("bnhrd, bncrd -> bnch", tmp_grd, percent)
                for i in range(n):
                    grad_arc[:, i : i + w] += arc_grd[:, i]

                # span_grd = arc_grd.sum([-1, -2])
                # diagonal_copy_(grad_span, span_grd, w)

            # prepare.
            closed_left = stripe(s_noninherent[..., LEFT], n, w - 1, (0, 1))
            closed_right = stripe(s_noninherent[..., RIGHT], n, w - 1, (1, w), 0)
            left_normalizer = stripe_with_headword(s_normalizer, n, w - 1, (0, 1))
            right_normalizer = stripe_with_headword(s_normalizer, n, w - 1, (1, w), 0)

            left = (
                (stripe_with_headword(s_inherent, n, w - 1, (0, 1)) + 1e-9).log_().add_(left_normalizer.unsqueeze(-1))
            )
            right = (
                (stripe_with_headword(s_inherent, n, w - 1, (1, w), 0) + 1e-9)
                .log_()
                .add_(right_normalizer.unsqueeze(-1))
            )

            # merge
            # forward pass. Prepare for all necessary quantities that are needed in the backward pass.
            headed = left.new_zeros(B, n, w - 1, w, r).fill_(-1e9)

            for i in range(w - 1):
                headed[:, :, i, : i + 1] = left[:, :, i, : i + 1] + closed_right[:, :, i, None, :]
                headed[:, :, i, i + 1 :] = right[:, :, i, i + 1 :] + closed_left[:, :, i, None, :]

            left_shape = left.shape
            closed_left_shape = closed_left.shape
            del left, right, closed_left, closed_right

            final_headed = headed.logsumexp(2)
            percent = (headed - final_headed.unsqueeze(2)).exp_()
            headed = final_headed
            del final_headed

            headed_normalizer = headed.max(-1)[0]
            headed = (headed - headed_normalizer.unsqueeze(-1)).exp()
            ## Backward pass. Estimate gradients.
            parent_grad_s = diagonal_with_headword(grad_s, w)

            # Backward head.
            tmp = torch.einsum("bnhm, bnhr -> bnhmr", parent_grad_s, headed)
            for i in range(n):
                grad_head[:, i : i + w, ...] += tmp[:, i]

            # Backward headed.
            rule = stripe_grammar_rules(head, n, w)
            tmp = torch.einsum("bnhm, bnhmr, bnhr, bnwhr -> bnwhr", parent_grad_s, rule, headed, percent)

            del rule

            # print("!!!", tmp)
            # tmp = tmp.scatter_(-1, index.unsqueeze(-1), (tmp.gather(-1, index.unsqueeze(-1)).squeeze(-1) - tmp.sum(-1)).unsqueeze(-1))
            # print("!!", tmp)
            # tmp = tmp[:, :, None, :] * percent
            # compute span marginal

            tmp_span_grd = tmp.sum([-1, -2])
            stripe_add_(grad_span, tmp_span_grd, n, w - 1, (0, 1))
            stripe_add_(grad_span, tmp_span_grd, n, w - 1, (1, w), 0)
            del tmp_span_grd

            grad_left = s.new_zeros(*left_shape)
            grad_right = s.new_zeros(*left_shape)
            grad_closed_left = s.new_zeros(*closed_left_shape)
            grad_closed_right = s.new_zeros(*closed_left_shape)

            for i in range(w - 1):
                grad_left[:, :, i, : i + 1] = tmp[:, :, i, : i + 1]
                grad_closed_right[:, :, i,] = (
                    tmp[:, :, i, : i + 1]
                ).sum(2)
                grad_right[:, :, i, i + 1 :] = tmp[:, :, i, i + 1 :]
                grad_closed_left[:, :, i,] = tmp[
                    :, :, i, i + 1 :
                ].sum(2)
                # grad_span_headword[:, :, i, :i + 1, :] = tmp[:, :, i, :i + 1]
                stripe_need_dad_add_(
                    grad_span_headword_left,
                    tmp[:, :, i, i + 1 :],
                    n,
                    w - (i + 1),
                    start=0,
                    end=i + 1,
                    headstart=i + 1,
                )
                stripe_need_dad_add_(
                    grad_span_headword_right, tmp[:, :, i, : i + 1], n, i + 1, start=i + 1, end=w, headstart=0
                )

            stripe_add_(grad_s_noninherent_l, grad_closed_left, n, w - 1, (0, 1))
            stripe_add_(grad_s_noninherent_r, grad_closed_right, n, w - 1, (1, w), 0)

            stripe_with_headword_add_(
                grad_s_inherent,
                (1 / (stripe_with_headword(s_inherent, n, w - 1, (0, 1)).clamp(min=1e-9))).mul_(grad_left),
                n,
                w - 1,
                (0, 1),
            )

            stripe_with_headword_add_(
                grad_s_inherent,
                (1 / (stripe_with_headword(s_inherent, n, w - 1, (1, w), 0).clamp(min=1e-9))).mul_(grad_right),
                n,
                w - 1,
                (1, w),
                0,
            )

            del grad_left, grad_right, tmp, grad_closed_left, grad_closed_right

        parent_grd = grad_s_inherent[:, torch.arange(H), torch.arange(H) + 1, torch.arange(H)]

        grad_inherent[:, t_slice] = parent_grd.sum(1).unsqueeze(1).expand(*grad_inherent[:, t_slice].shape)

        parent_grd = torch.cat(
            [
                grad_s_noninherent_l[:, torch.arange(H), torch.arange(H) + 1].unsqueeze(-1),
                grad_s_noninherent_r[:, torch.arange(H), torch.arange(H) + 1].unsqueeze(-1),
            ],
            dim=-1,
        )

        grad_noninherent_word += parent_grd.sum(-1)

        grad_noninherent_symbol[:, t_slice] = (
            ((1 / (noninherent_symbol[:, t_slice, :, :].sum(1)).clamp(min=1e-9))[:, None, ...] * parent_grd)
            .sum(1)
            .unsqueeze(1)
            .expand(*grad_noninherent_symbol[:, t_slice].shape)
        )

        tmp_grd = torch.cat(
            [diagonal(grad_span_headword_left, 1).unsqueeze(-1), diagonal(grad_span_headword_right, 1).unsqueeze(-1)],
            dim=-1,
        )

        grad_arc += tmp_grd.sum([-1, -2])

        # grad_inherent,grad_noninherent_symbol, grad_noninherent_word, grad_span, grad_head, grad_arc

        ctx.grad_inherent = grad_inherent
        ctx.grad_noninherent_symbol = grad_noninherent_symbol
        ctx.grad_noninherent_word = grad_noninherent_word
        ctx.grad_head = grad_head
        ctx.grad_root = grad_root
        ctx.arc_marginal = grad_arc
        ctx.span_marginal = grad_span
        return logZ

    @staticmethod
    def backward(ctx, grad_output):
        multiplier = grad_output.max()
        return (
            ctx.grad_noninherent_word * multiplier,
            ctx.grad_noninherent_symbol * multiplier,
            ctx.grad_inherent * multiplier,
            ctx.grad_root * multiplier,
            ctx.grad_head * multiplier,
            ctx.span_marginal,
            ctx.arc_marginal,
            None,
        )


class Operation1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, left_closed, right_closed):
        batch, n, w, *_, r = left.shape
        headed = left.new_zeros(batch, n, w, w + 1, r).fill_(-1e9)
        for i in range(w):
            headed[:, :, i, : i + 1] = left[:, :, i, : i + 1] + right_closed[:, :, i, None, :]
            headed[:, :, i, i + 1 :] = right[:, :, i, i + 1 :] + left_closed[:, :, i, None, :]
        ctx.shape1 = left.shape
        ctx.shape2 = left_closed.shape
        del left, right, left_closed, right_closed
        final_headed = headed.logsumexp(2)
        ctx.percent = (headed - final_headed.unsqueeze(2)).exp_()
        return final_headed

    @staticmethod
    def backward(ctx, grad_output):
        shape1 = ctx.shape1
        shape2 = ctx.shape2
        percent = ctx.percent
        tmp = grad_output[:, :, None, :] * percent
        del ctx.percent
        grad_left = grad_output.new_zeros(*shape1)
        grad_right = grad_output.new_zeros(*shape1)
        grad_closed_left = grad_output.new_zeros(*shape2)
        grad_closed_right = grad_output.new_zeros(*shape2)
        w = shape1[2]
        for i in range(w):
            grad_left[:, :, i, : i + 1] = tmp[:, :, i, : i + 1]
            grad_closed_right[:, :, i] = tmp[:, :, i, : i + 1].sum(2)
            grad_right[:, :, i, i + 1 :] = tmp[:, :, i, i + 1 :]
            grad_closed_left[:, :, i] = tmp[:, :, i, i + 1 :].sum(2)
        return grad_left, grad_right, grad_closed_left, grad_closed_right
