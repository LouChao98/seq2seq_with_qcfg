from typing import Dict, List, Union

import numpy as np
import torch
from numba import jit, prange
from torch import Tensor
from torch.autograd import grad

from ._utils import checkpoint, weighted_random
import torch_semiring_einsum as tse


class D1PCFG:
    # A[i] -> B[j], C[k]
    # ================
    # A[i] -> R
    # R -> B
    # R -> C
    # R, i -> j
    # R, i -> k
    # ================
    # Time complexity: 6

    def __init__(self, tgt_nt_states, tgt_pt_states) -> None:
        self.tgt_nt_states = tgt_nt_states
        self.tgt_pt_states = tgt_pt_states
        self.block_size = 32

        self.eq_slr = tse.compile_equation("qrij, qrik->qrijk")
        self.eq_qnkrj = tse.compile_equation("qnwjr,qnwkr->qnkrj")
        self.eq_qnri = tse.compile_equation("qnkrj,qrijk->qnri")
        self.eq_qnai = tse.compile_equation("qnri,qair->qnai")
        self.eq_tor = tse.compile_equation("xlpi,xrp->xlir")

    def __call__(self, params: Dict[str, Tensor], lens, decode=False, marginal=False):
        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            # NOTE I assume marginals are only used for decoding.
            params = {k: v.detach() for k, v in params.items()}
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens)
        assert (
            lens[1:] <= lens[:-1]
        ).all(), "You should sort samples by length descently."

        terms = params["term"]  # (batch, seq_len, PT)
        root = params["root"]  # (batch, NT)

        batch, N, PT = terms.shape
        N += 1
        NT = root.shape[1]
        nt_spans = NT // self.tgt_nt_states
        pt_spans = PT // self.tgt_pt_states

        terms = terms.view(batch, -1, self.tgt_pt_states, pt_spans)
        root = root.view(batch, self.tgt_nt_states, nt_spans)

        # {source,target}{left,right}[{nonterminal,preterminal}]
        H = params["head"]  # (batch, NT, r), A[i] -> R
        # (batch, r, TGT_NT), R -> B
        # (batch, r, TGT_PT), R -> B
        TLNT, TLPT = torch.split(
            params["left"], (self.tgt_nt_states, self.tgt_pt_states), -1
        )
        # (batch, r, TGT_NT), R -> C
        # (batch, r, TGT_PT), R -> C
        TRNT, TRPT = torch.split(
            params["right"], (self.tgt_nt_states, self.tgt_pt_states), -1
        )
        R = H.shape[-1]
        H = H.view(batch, self.tgt_nt_states, nt_spans, R)

        # ===== This routine is optimized for src_len < tgt_len =====

        if "slr" in params:
            SLR = params["slr"]
        else:
            SL = params["sl"]  # (batch, r, SRC_NT, SRC_NT), R, i -> j
            SR = params["sr"]  # (batch, r, SRC_NT, SRC_NT), R, i -> k
            SLR = tse.log_einsum(self.eq_slr, SL, SR, block_size=self.block_size)

        def merge_without_indicator(Y, Z):
            qnkrj = tse.log_einsum(self.eq_qnkrj, Y, Z, block_size=self.block_size)
            qnri = tse.log_einsum(self.eq_qnri, qnkrj, SLR, block_size=self.block_size)
            qnai = tse.log_einsum(self.eq_qnai, qnri, H, block_size=self.block_size)
            return qnai

        def merge_with_indicator(Y, Z, indicator):
            qnkrj = tse.log_einsum(self.eq_qnkrj, Y, Z, block_size=self.block_size)
            qnri = tse.log_einsum(self.eq_qnri, qnkrj, SLR, block_size=self.block_size)
            qnai = tse.log_einsum(self.eq_qnai, qnri, H, block_size=self.block_size)
            qnai = qnai + indicator
            return qnai

        # ===== End =====

        left_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        right_s = terms.new_full((batch, N, N, nt_spans, R), -1e9)
        left_term = tse.log_einsum(self.eq_tor, terms, TLPT, block_size=self.block_size)
        right_term = tse.log_einsum(
            self.eq_tor, terms, TRPT, block_size=self.block_size
        )
        diagonal_copy_(left_s, left_term, w=1)
        diagonal_copy_(right_s, right_term, w=1)

        if marginal:
            span_indicator = terms.new_zeros(batch, N, N).requires_grad_()
            span_indicator_running = span_indicator[:]
        else:
            span_indicator = None

        # prepare length, same as the batch_size in PackedSequence
        n_at_position = (
            torch.arange(2, N + 1).unsqueeze(1) <= lens.cpu().unsqueeze(0)
        ).sum(1)

        # w: span width
        final = []
        for step, w in enumerate(range(2, N)):

            # n: the number of spans of width w.
            n = N - w
            y = stripe(left_s, n, w - 1, (0, 1))
            z = stripe(right_s, n, w - 1, (1, w), 0)
            if marginal:
                x = merge_with_indicator(
                    y.clone(),
                    z.clone(),
                    span_indicator_running.diagonal(w, 1, 2)
                    .unsqueeze(-1)
                    .unsqueeze(-1),
                )
            else:
                x = merge_without_indicator(y.clone(), z.clone())

            unfinished = n_at_position[step + 1]
            current_bsz = n_at_position[step]

            if current_bsz - unfinished > 0:
                final.insert(0, x[unfinished:current_bsz, :1])
            if unfinished > 0:
                x = x[:unfinished]
                left_s = left_s[:unfinished]
                right_s = right_s[:unfinished]
                SLR = SLR[:unfinished]
                H = H[:unfinished]
                TLNT = TLNT[:unfinished]
                TRNT = TRNT[:unfinished]
                if marginal:
                    span_indicator_running = span_indicator_running[:unfinished]

                left_x = tse.log_einsum(
                    self.eq_tor, x, TLNT, block_size=self.block_size
                )
                right_x = tse.log_einsum(
                    self.eq_tor, x, TRNT, block_size=self.block_size
                )
                diagonal_copy_(left_s, left_x, w)
                diagonal_copy_(right_s, right_x, w)

        final = torch.cat(final, dim=0)

        final = final.squeeze(1) + root
        logZ = final.logsumexp((-2, -1))
        if decode:
            spans = self.get_prediction(logZ, span_indicator, lens)
            spans = [[(span[0], span[1] - 1, 0) for span in inst] for inst in spans]
            return spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            return grad(logZ.sum(), [span_indicator])[0]
        return -logZ

    @torch.no_grad()
    def sampled_decoding(
        self,
        params,
        nt_spans,
        src_nt_states,
        pt_spans,
        src_pt_states,
        use_copy=True,
        num_samples=10,
        max_length=100,
    ):
        terms: torch.Tensor = params["term"].detach()
        roots: torch.Tensor = params["root"].detach()
        H: torch.Tensor = params["head"].detach()  # (batch, NT, r) r:=rank
        L: torch.Tensor = params["left"].detach()  # (batch, r, TGT_NT + TGT_PT)
        R: torch.Tensor = params["right"].detach()  # (batch, r, TGT_NT + TGT_PT)
        SLR: torch.Tensor = params["slr"].detach()

        zero = terms.new_full((1,), -1e9)
        threshold = terms.new_full((1,), np.log(1e-2))
        terms = torch.where(terms > threshold, terms, zero).softmax(2).cumsum(2)
        roots = torch.where(roots > threshold, roots, zero).softmax(1).cumsum(1)
        H = torch.where(H > threshold, H, zero).softmax(2).cumsum(2)
        L = torch.where(L > threshold, L, zero).softmax(2).cumsum(2)
        R = torch.where(R > threshold, R, zero).softmax(2).cumsum(2)
        SLR = (
            torch.where(SLR > threshold, SLR, zero)
            .view(*SLR.shape[:2], -1)
            .softmax(2)
            .cumsum(2)
        )

        terms[..., -1] += 1  # avoid out of bound
        roots[..., -1] += 1
        H[..., -1] += 1
        L[..., -1] += 1
        R[..., -1] += 1
        SLR[..., -1] += 1

        terms = terms.cpu().numpy()
        roots = roots.cpu().numpy()
        H = H.cpu().numpy()
        L = L.cpu().numpy()
        R = R.cpu().numpy()
        SLR = SLR.cpu().numpy()

        preds = []
        for b in range(len(terms)):
            samples, scores = self.sample(
                terms[b],
                H[b],
                L[b],
                R[b],
                SLR[b],
                roots[b],
                nt_spans[b],
                src_nt_states,
                pt_spans[b],
                src_pt_states,
                use_copy=use_copy,
                num_samples=num_samples,
                max_length=max_length,
            )
            sample_scores = [(sample, score) for sample, score in zip(samples, scores)]
            sample_scores.sort(key=lambda x: x[1], reverse=True)
            preds.append(sample_scores)
        return preds

    @jit(nopython=True, parallel=True)
    def sample(
        self,
        terms: np.ndarray,  # seqlen x pt, in normal space
        rules_head: np.ndarray,  # nt x r, in normal space
        rules_left: np.ndarray,  # (nt+pt) x r, in normal space
        rules_right: np.ndarray,  # (nt+pt) x r, in normal space
        rules_src: np.ndarray,  # src x src x src, in normal space
        roots: np.ndarray,  # nt, in normal space
        nt_spans: List[List[str]],
        nt_states: int,
        pt_spans: List[List[str]],
        pt_states: int,
        use_copy=True,
        num_samples=1,
        max_length=100,
        max_actions=100,
        UNK=1,
    ):
        # TODO
        NT = rules_head.shape[0]
        COPY_NT = nt_states - 1
        COPY_PT = pt_states - 1
        samples = [None for _ in range(num_samples)]
        scores = [None for _ in range(num_samples)]
        nt_num_nodes = len(nt_spans)
        pt_num_nodes = len(pt_spans)

        for i in prange(num_samples):
            actions = 0
            sample = weighted_random(roots)
            score = roots[sample]
            nonterminals: List[int] = [sample]
            preterminals: List[Union[List[str], int]] = []

            while (
                len(nonterminals) > 0
                and len(preterminals) < max_length
                and actions < max_actions
            ):
                s = nonterminals.pop()
                if s < NT:
                    if use_copy:
                        nt_state = s // nt_num_nodes
                        if nt_state == COPY_NT:
                            nt_node = s % nt_num_nodes
                            preterminals.append(nt_spans[nt_node])
                    else:
                        actions += 1
                        head = weighted_random(rules_head[s])
                        left = weighted_random(rules_left[head])
                        right = weighted_random(rules_right[head])
                        score += (
                            rules_head[s, head]
                            + rules_left[left, head]
                            + rules_right[right, head]
                        )
                        nonterminals.extend([left, right])
                else:
                    preterminals.append(s - NT)

            preterminals = preterminals[::-1]
            terminals: List[Union[str, int]] = []
            for s in preterminals:
                if isinstance(s, list):
                    terminals.extend(s)
                else:
                    src_pt_state = s // pt_num_nodes
                    if use_copy and src_pt_state == COPY_PT:
                        src_node = s % pt_num_nodes
                        terminals.extend(pt_spans[src_node])
                    else:
                        sample = weighted_random(terms[s])
                        score += terms[s, sample]
                        if use_copy and sample == UNK:
                            # force <unk> tokens to copy
                            src_node = s % pt_num_nodes
                            terminals.extend(pt_spans[src_node])
                        else:
                            terminals.append(sample)
            samples[i] = terminals
            scores[i] = score / len(terminals)
        return samples, scores

    def get_prediction(self, logZ, span_indicator, lens):
        batch, seq_len = span_indicator.shape[:2]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            marginals = grad(logZ.sum(), [span_indicator])[0].detach()
            return self._cky_zero_order(marginals.detach(), lens)
        else:
            # minimal length is 2
            predictions = [[(0, 0), (1, 1), (0, 1)] for _ in range(batch)]
            return predictions

    @torch.no_grad()
    def _cky_zero_order(self, marginals, lens):
        N = marginals.shape[-1]
        s = marginals.new_zeros(*marginals.shape).fill_(-1e9)
        p = marginals.new_zeros(*marginals.shape).long()
        diagonal_copy_(s, diagonal(marginals, 1), 1)
        for w in range(2, N):
            n = N - w
            starts = p.new_tensor(range(n))
            if w != 2:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            else:
                Y = stripe(s, n, w - 1, (0, 1))
                Z = stripe(s, n, w - 1, (1, w), 0)
            X, split = (Y + Z).max(2)
            x = X + diagonal(marginals, w)
            diagonal_copy_(s, x, w)
            diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)

        def backtrack(p, i, j):
            if j == i + 1:
                return [(i, j)]
            split = p[i][j]
            ltree = backtrack(p, i, split)
            rtree = backtrack(p, split, j)
            return [(i, j)] + ltree + rtree

        p = p.tolist()
        lens = lens.tolist()
        spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
        for spans_inst in spans:
            spans_inst.sort(key=lambda x: x[1] - x[0])
        return spans

    def convert_to_tree(self, spans, length):
        tree = [(i, str(i)) for i in range(length)]
        tree = dict(tree)
        for l, r, _ in spans:
            if l != r:
                span = "({} {})".format(tree[l], tree[r])
                tree[r] = tree[l] = span
        return tree[0]


def diagonal_copy_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
            stride=new_stride,
            storage_offset=w * stride[2],
        ).copy_(y)
    else:
        x.as_strided(
            size=(x.shape[0], seq_len - w),
            stride=new_stride,
            storage_offset=w * stride[2],
        ).copy_(y)


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(
            size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
            stride=new_stride,
            storage_offset=w * stride[2],
        )
    else:
        return x.as_strided(
            size=(x.shape[0], seq_len - w),
            stride=new_stride,
            storage_offset=w * stride[2],
        )


def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(
            size=(x.shape[0], n, w, *list(x.shape[3:])),
            stride=stride,
            storage_offset=(offset[0] * seq_len + offset[1]) * numel,
        )
    else:
        return x.as_strided(
            size=(x.shape[0], n, w),
            stride=stride,
            storage_offset=(offset[0] * seq_len + offset[1]) * numel,
        )


if __name__ == "__main__":
    from .pcfg import PCFG

    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(1)
    B, N, TGT_PT, SRC_PT, TGT_NT, SRC_NT, r = 4, 5, 3, 3, 3, 3, 2
    NT = TGT_NT * SRC_NT
    T = TGT_PT * SRC_PT
    device = "cuda"

    slr = (
        torch.randn(B, r, SRC_NT, SRC_NT, SRC_NT, device=device)
        .view(B, r * SRC_NT, -1)
        .log_softmax(-1)
        .view(B, r, SRC_NT, SRC_NT, SRC_NT)
    )
    params = {
        "term": torch.randn(B, N, T, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "root": torch.randn(B, NT, device=device).log_softmax(-1).requires_grad_(True),
        "head": torch.randn(B, NT, r, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "left": torch.randn(B, r, TGT_NT + TGT_PT, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "right": torch.randn(B, r, TGT_NT + TGT_PT, device=device)
        .log_softmax(-1)
        .requires_grad_(True),
        "slr": slr,
    }
    lens = torch.tensor([N, N - 1, N - 1, N - 3], dtype=torch.long, device=device)

    pcfg = D1PCFG(TGT_NT, TGT_PT)

    print(pcfg(params, lens))
    m1 = pcfg(params, lens, marginal=True)
    print(m1.sum((1, 2)))

    head = params["head"].view(B, TGT_NT, SRC_NT, r)
    rule = torch.einsum(
        "xair,xrb,xrc,xrijk->xaibjck",
        head.exp(),
        params["left"].exp(),
        params["right"].exp(),
        params["slr"].exp(),
    )
    shape = rule.shape
    rule = rule.reshape(
        shape[0], shape[1] * shape[2], shape[3] * shape[4], shape[5] * shape[6]
    ).log()

    params2 = {"term": params["term"], "rule": rule, "root": params["root"]}
    pcfg2 = PCFG()
    print(pcfg2(params2, lens))

