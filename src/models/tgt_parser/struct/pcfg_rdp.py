import numpy as np
import torch
from torch import Tensor
from torch.autograd import grad

from ._fn import diagonal, diagonal_copy_, stripe
from ._utils import checkpoint, process_param_for_marginal
from .pcfg import PCFG
from .td_pcfg import FastestTDPCFG


class PCFGRandomizedDP(FastestTDPCFG):
    """This only implement inside, marginal and decode.
    For sampling, use naive PCFG."""

    def __init__(self, topk, sample_size, smooth):
        self.topk = topk
        self.sample_size = sample_size
        self.smooth = smooth

    def __call__(self, params, lens, decode=False, marginal=False):
        # terms: bsz x seqlen x pt
        # rules: bsz x nt x (nt+pt) x (nt+pt)
        # roots: bsz x nt

        topk, sample_size, smooth = self.topk, self.sample_size, self.smooth
        batch, N, T = params["term"].shape
        N += 1
        NT = params["rule"].shape[1]
        S = NT + T
        RS = topk + sample_size

        if RS >= NT:
            return PCFG()(params, lens, decode, marginal)

        if decode:
            marginal = True  # MBR decoding
        if marginal:
            grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            cm = torch.inference_mode(False)
            cm.__enter__()
            params = {k: process_param_for_marginal(v) for k, v in params.items()}

        terms, rules, roots = params["term"], params["rule"], params["root"]
        copy_nts = params.get("copy_nt")

        s = terms.new_full((batch, N, N, RS), -1e9)
        s_ind = terms.new_zeros(batch, N, N, RS, dtype=torch.long)

        NTs = slice(0, NT)
        Ts = slice(NT, S)
        rXYZ = rules[:, :, NTs, NTs].contiguous()
        rXyZ = rules[:, :, Ts, NTs].contiguous()
        rXYz = rules[:, :, NTs, Ts].contiguous()
        rXyz = rules[:, :, Ts, Ts].contiguous()

        span_indicator = rules.new_zeros(batch, N, N, 1, requires_grad=marginal)

        for step, w in enumerate(range(2, N)):
            n = N - w

            Y_term = terms[:, :n, :, None]
            Z_term = terms[:, w - 1 :, None, :]
            indicator = span_indicator.diagonal(w, 1, 2).movedim(-1, 1)

            if w == 2:
                score = Xyz(Y_term, Z_term, rXyz)
                score, ind = sample(score, topk, sample_size, smooth)
                diagonal_copy_(s, score + indicator, w)
                diagonal_copy_(s_ind, ind, w)
                continue

            x = terms.new_zeros(3, batch, n, NT).fill_(-1e9)

            Y = stripe(s, n, w - 1, (0, 1)).clone()
            Y_ind = stripe(s_ind, n, w - 1, (0, 1)).clone()  # why need clone here
            Z = stripe(s, n, w - 1, (1, w), 0).clone()
            Z_ind = stripe(s_ind, n, w - 1, (1, w), 0).clone()

            if w > 3:
                x[0].copy_(XYZ(Y, Y_ind, Z, Z_ind, rXYZ))

            x[1].copy_(XYz(Y, Y_ind, Z_term, rXYz))
            x[2].copy_(XyZ(Y_term, Z, Z_ind, rXyZ))
            x = x.logsumexp(0)

            x, ind = sample(x, topk, sample_size, smooth)

            if copy_nts is not None:
                value, mask = copy_nts[step]
                value = value.gather(2, ind)
                mask = mask.gather(2, ind)
                x = torch.where(mask, value, x)

            diagonal_copy_(s, x + indicator, w)
            diagonal_copy_(s_ind, ind, w)

        b_ind = torch.arange(batch, device=roots.device)
        if isinstance(lens, list):
            lens = torch.tensor(lens)
        lens = lens.to(roots.device)
        root_ind = s_ind[b_ind, 0, lens]
        roots = roots.gather(1, root_ind)
        logZ = (s[b_ind, 0, lens] + roots).logsumexp(-1)

        if decode:
            spans = self.mbr_decoding(logZ, span_indicator, lens)
            # spans = [[(span[0], span[1] - 1, 0) for span in inst] for inst in spans]
            return spans
            # trees = []
            # for spans_inst, l in zip(spans, lens.tolist()):
            #     tree = self.convert_to_tree(spans_inst, l)
            #     trees.append(tree)
            # return trees, spans
        if marginal:
            torch.set_grad_enabled(grad_state)
            cm.__exit__(None, None, None)
            return grad(logZ.sum(), [span_indicator])[0]
        return -logZ


def sample(score: Tensor, topk: int, sample: int, smooth: float):
    # I use normalized scores p(l|i,j) as the proposal distribution

    # Get topk for exact marginalization
    b, n, c = score.shape
    proposal_p = score.detach().softmax(-1)
    _, topk_ind = torch.topk(proposal_p, topk, dim=-1, sorted=False)
    topk_score = score.gather(-1, topk_ind)

    if sample == 0:
        return topk_score, topk_ind

    # get renormalized proposal distribution
    b_ind = torch.arange(b, device=score.device)
    b_ind = b_ind.unsqueeze(-1).expand(b, n * topk).flatten()
    n_ind = torch.arange(n, device=score.device)
    n_ind = n_ind.view(1, n, 1).expand(b, n, topk).flatten()
    proposal_p += smooth
    proposal_p[b_ind, n_ind, topk_ind.flatten()] = 0
    proposal_p /= proposal_p.sum(-1, keepdim=True) + 1e-6

    # sample from the proposal.
    sampled_ind = torch.multinomial(proposal_p.view(-1, c), sample).view(b, n, sample)
    sample_log_prob = proposal_p.gather(-1, sampled_ind)
    sample_log_prob = (sample_log_prob + 1e-8).log()
    sampled_score = score.gather(-1, sampled_ind)

    # debias sampled emission
    sampled_ind = sampled_ind.view(b, n, sample)
    sample_log_prob = sample_log_prob.view(b, n, sample)
    correction = sample_log_prob + np.log(sample)
    sampled_score -= correction

    # Combine the emission
    combined_score = torch.cat([topk_score, sampled_score], dim=-1)
    combined_ind = torch.cat([topk_ind, sampled_ind], dim=-1)
    return combined_score, combined_ind


@checkpoint
def Xyz(y: Tensor, z: Tensor, rule: Tensor):
    b, n, t, _ = y.shape
    b_n_yz = (y + z).view(*y.shape[:2], -1)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule.view(b, 1, -1, t * t)).logsumexp(-1)
    return b_n_x


@checkpoint
def XYZ(Y: Tensor, Y_ind: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    b, n, w, nt = Y.shape
    w -= 2
    NT = rule.shape[1]

    ind = Y_ind[:, :, 1:-1, :, None] * NT + Z_ind[:, :, 1:-1, None, :]
    ind = ind.flatten(3)[:, :, :, None].expand(-1, -1, -1, NT, -1)
    rule = rule.view(b, 1, 1, NT, -1).expand(-1, n, w, -1, -1)
    rule = rule.gather(4, ind).view(b, n, w, NT, nt, nt)

    # Y_ind = Y_ind[:, :, 1:-1, None, :, None].expand(-1, -1, -1, NT, -1, NT)
    # Z_ind = Z_ind[:, :, 1:-1, None, None, :].expand(-1, -1, -1, NT, nt, -1)
    # rule1 = rule.view(b, 1, 1, NT, NT, NT).expand(-1, n, w, -1, -1, -1)
    # rule1 = rule1.gather(4, Y_ind).gather(5, Z_ind)

    b_n_yz = Y[:, :, 1:-1, :].unsqueeze(-1) + Z[:, :, 1:-1, :].unsqueeze(-2)
    b_n_x = (b_n_yz.unsqueeze(3) + rule).logsumexp((2, 4, 5))
    return b_n_x


@checkpoint
def XYz(Y: Tensor, Y_ind: Tensor, z: Tensor, rule: Tensor):
    b, n, _, nt = Y.shape
    t = z.shape[-1]
    Y = Y[:, :, -1, :, None]
    Y_ind = (
        Y_ind[:, :, -1]
        .view(b, n, 1, -1, 1)
        .expand(-1, -1, rule.shape[1], -1, rule.shape[3])
    )
    rule = rule.unsqueeze(1).expand(-1, n, -1, -1, -1)
    rule = rule.gather(3, Y_ind).flatten(3)
    b_n_yz = (Y + z).reshape(b, n, nt * t)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule).logsumexp(-1)
    return b_n_x


@checkpoint
def XyZ(y: Tensor, Z: Tensor, Z_ind: Tensor, rule: Tensor):
    b, n, _, nt = Z.shape
    t = y.shape[-2]
    Z = Z[:, :, 0, None, :]
    Z_ind = (
        Z_ind[:, :, 0]
        .view(b, n, 1, 1, -1)
        .expand(-1, -1, rule.shape[1], rule.shape[2], -1)
    )
    rule = rule.unsqueeze(1).expand(-1, n, -1, -1, -1)
    rule = rule.gather(4, Z_ind).flatten(3)
    b_n_yz = (y + Z).reshape(b, n, nt * t)
    b_n_x = (b_n_yz.unsqueeze(-2) + rule).logsumexp(-1)
    return b_n_x
