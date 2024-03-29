import inspect
import logging
from ctypes import ArgumentError
from enum import IntEnum
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import grad
from torch.distributions.utils import lazy_property

from ._fn import ns_diagonal as diagonal
from ._fn import ns_diagonal_copy_ as diagonal_copy_
from ._fn import ns_stripe as stripe
from .semiring import CrossEntropySemiring, EntropySemiring, KLSemiring, LogSemiring, MaxSemiring, SampledSemiring

log = logging.getLogger(__file__)
_OK, _SONMASK, _REACHLIMIT = 0, 1, 2
_VOCAB, _COPY_NT, _COPY_PT = 0, 1, 2


class TokenType(IntEnum):
    VOCAB = _VOCAB
    COPY_NT = _COPY_NT
    COPY_PT = _COPY_PT


class DecompBase:
    KEYS: Optional[List[str]] = None
    LOGSPACE: Optional[List[bool]] = None
    FORCE_ORDERED: bool = True

    def __init__(
        self,
        params: Dict[str, Tensor],
        lens,
        nt_states,
        nt_num_nodes,
        pt_states,
        pt_num_nodes,
        batch_size,
        no_trace=False,
    ):

        super().__init__()
        self.params = params
        self.lens = lens = torch.tensor(lens)
        self.batch_size = batch_size
        self.nt_states = nt_states
        self.nt_num_nodes = nt_num_nodes
        self.pt_states = pt_states
        self.pt_num_nodes = pt_num_nodes
        self.max_states = max(nt_states, pt_states)
        self.no_trace = no_trace

        self.is_ordered = (lens[1:] <= lens[:-1]).all()
        assert not self.FORCE_ORDERED or self.is_ordered, "Expect lengths in descending."

        self._traced_cache = None
        self._untraced_cache = None

    def __call__(self, trace=None) -> Tuple[Tensor, Optional[Tensor]]:
        # share computation between self.partition, self.nll
        if trace is None:
            trace = not self.no_trace and torch.is_grad_enabled()
        if self._traced_cache is not None:
            return self._traced_cache
        if not trace and self._untraced_cache is not None:
            return self._untraced_cache
        result = self.inside(self.params, LogSemiring, trace)
        if trace:
            self._traced_cache = result
        else:
            self._untraced_cache = result
        return result

    def inside(self, params, semiring, trace=False) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def preprocess(self, params, semiring) -> Dict[str, Any]:
        # combine params in self.KEYS. set others using params[0] if there are more than 1.

        if isinstance(params, tuple):
            extra_params = set(params[0].keys()) - set(self.KEYS)
            new_params = {k: semiring.convert(params[0][k], params[1][k]) for k in self.KEYS}
            for key in extra_params:
                if key == "constraint":
                    if params[0]["constraint"] is not None:
                        converted = []
                        for (value, mask), (value2, mask2) in zip(params[0]["constraint"], params[1]["constraint"]):
                            converted.append((semiring.convert(value, value2), mask))
                        new_params[key] = converted
                    else:
                        assert params[1].get("constraint") is None
                elif key == "add":
                    if params[0].get(key) is not None or params[1].get(key) is not None:
                        converted = []

                        if params[0].get(key) is None:
                            params[0][key] = [torch.zeros_like(item) for item in params[1][key]]
                        elif params[1].get(key) is None:
                            params[1][key] = [torch.zeros_like(item) for item in params[0][key]]

                        for value1, value2 in zip(params[0][key], params[1][key]):
                            converted.append(semiring.convert(value1, value2))
                        new_params[key] = converted
                elif key.startswith("_"):
                    continue
                else:
                    assert params[0][key] is None, f"Not implemented for {key}"
            semiring.set_device(params[0][self.KEYS[0]].device)
        else:
            extra_params = set(params.keys()) - set(self.KEYS)
            new_params = {k: semiring.convert(params[k]) for k in self.KEYS}
            for key in extra_params:
                if key == "constraint":
                    if params["constraint"] is not None:
                        converted = []
                        for value, mask in params["constraint"]:
                            converted.append((semiring.convert(value), mask))
                        new_params[key] = converted
                elif key in ("add", "lse"):
                    if params[key] is not None:
                        converted = []
                        for value in params[key]:
                            converted.append(semiring.convert(value))
                        new_params[key] = converted
                elif key.startswith("_"):
                    continue
                else:
                    assert params[key] is None, f"Not implemented for {key}"
            semiring.set_device(params[self.KEYS[0]].device)

        return new_params

    @property
    def partition(self):
        # do not use lazy_property. I have cached the result.
        return self()[0]

    def partition_at_length(self, params, length):
        # need unprocessed params TODO refactor this
        # terms: bsz x (tgt pt x src pt) x vocab
        if isinstance(length, list):
            length = max(length)
        term = LogSemiring.sum(params["term"], dim=2).unsqueeze(1).expand(-1, length, -1)
        params = {**params} | {"term": term}
        return self.inside(params, LogSemiring)[0]

    @property
    def nll(self):
        # do not use lazy_property. I have cached the result.
        return -self.partition

    def score(self, event):
        # event: dict containing 0/1 indicator of rules
        output = 0
        for k, p in self.log_params.items():
            e = event[k]
            output += (p * e).flatten(1).sum(1)
        return output

    @lazy_property
    def marginal(self):
        params = {}
        for key, value in self.params.items():
            if key in self.KEYS:
                params[key] = value.detach().requires_grad_()
            else:
                params[key] = value
        logZ, trace = self.inside(params, LogSemiring, trace=True)
        logZ.sum().backward()
        output = {}
        for k, is_in_log_space in zip(self.KEYS, self.LOGSPACE):
            g = params[k].grad
            if is_in_log_space:
                output[k] = g
            else:
                output[k] = g * params[k].detach()
        output["trace"] = trace.grad
        return output

    @lazy_property
    def marginal_with_grad(self):
        params = {}
        # TODO only check term should be enough
        for key, value in self.params.items():
            if key in self.KEYS and not value.requires_grad:
                params[key] = value.requires_grad_()
            else:
                params[key] = value
        logZ, trace = self.inside(self.params, LogSemiring, trace=True, use_reentrant=False)
        grads = grad(logZ.sum(), [self.params["term"], trace], create_graph=True)
        return grads

    @lazy_property
    def rule_marginal_with_grad(self):
        params = {}
        for key, value in self.params.items():
            if key in self.KEYS and not value.requires_grad:
                params[key] = value.requires_grad_()
            else:
                params[key] = value
        logZ, trace = self.inside(self.params, LogSemiring, trace=False, use_reentrant=False)
        grads = grad(logZ.sum(), [self.params[key] for key in self.KEYS], create_graph=True)
        output = {}
        for k, is_in_log_space, g in zip(self.KEYS, self.LOGSPACE, grads):
            if is_in_log_space:
                output[k] = g
            else:
                output[k] = g * self.params[k]
        return output

    @lazy_property
    def entropy(self):
        return self.inside(self.params, EntropySemiring, False)[0]

    def cross_entropy(self, other: "DecompBase", fix_left=False):
        if fix_left:
            sparams = {k: v.detach() if isinstance(v, Tensor) else v for k, v in self.params.items()}
        else:
            sparams = self.params
        return self.inside((sparams, other.params), CrossEntropySemiring, False)[0]

    def kl(self, other: "DecompBase", fix_left=False):
        if fix_left:
            sparams = {k: v.detach() if isinstance(v, Tensor) else v for k, v in self.params.items()}
        else:
            sparams = self.params
        return self.inside((sparams, other.params), KLSemiring, False)[0]

    @torch.enable_grad()
    def sample_one(self, need_event=False, need_span=True):
        params = {
            k: v.detach().requires_grad_() if isinstance(v, torch.Tensor) else v for k, v in self.params.items()
        }
        logZ, trace = self.inside(params, SampledSemiring, True)
        logZ.sum().backward()

        output = {"logZ": logZ}

        if need_span:
            spans = [[] for _ in range(self.batch_size)]
            for b, i, state_node in params["term"].grad.nonzero().tolist():
                state, node = divmod(state_node, self.pt_num_nodes)
                spans[b].append((i, i + 1, state, node))
            for b, i, j, state, node in trace.grad.nonzero().tolist():
                spans[b].append((i, j, state, node))
            for spans_item in spans:
                spans_item.sort(key=lambda x: (x[1] - x[0], x[0]))
            output["span"] = spans

        if need_event:
            grads = {}
            for k, is_in_log_space in zip(self.KEYS, self.LOGSPACE):
                if is_in_log_space:
                    grads[k] = params[k].grad.detach()
                else:
                    grads[k] = (params[k].grad * self.params[k]).detach()

            output["event"] = grads | {"trace": trace.grad}

        return output

    def gumbel_sample_one(self, temperature):
        raise NotImplementedError

    def argmax_st(self):
        raise NotImplementedError

    @property
    def decoded(self):
        try:
            spans = self.viterbi_decoded
        except NotImplementedError:
            spans = self.mbr_decoded

        for spans_item in spans:
            spans_item.sort(key=lambda x: (x[1] - x[0], x[0]))
        return spans

    @lazy_property
    def viterbi_decoded(self):
        raise NotImplementedError

    @lazy_property
    def mbr_decoded(self):
        if self.params["term"].ndim == 3:
            return self.mbr_decoding_1gram_pt()
        else:
            return self.mbr_decoding_ngram_pt()

    def mbr_decoding_1gram_pt(self):
        # term: b, n, tgt_pt, src_pt
        # trace: b, n+1, n+1, tgt_nt, src_nt
        batch, seq_len = self.params["term"].shape[:2]
        marginal = self.marginal
        trace_m: Tensor = marginal["trace"]
        term_m: Tensor = marginal["term"]
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            scores = trace_m.flatten(3).sum(-1) if trace_m.ndim > 3 else trace_m
            spans = _cky_zero_order(scores, self.lens)
        else:
            spans = [[(0, 1), (1, 2), (0, 2)] for _ in range(batch)]

        has_label = trace_m.ndim > 3
        pt_label = term_m.flatten(2).argmax(-1)
        pt_label_tgt = torch.div(pt_label, self.pt_num_nodes, rounding_mode="floor")
        pt_label_src = torch.remainder(pt_label, self.pt_num_nodes)
        pt_label_tgt = pt_label_tgt.cpu().numpy()
        pt_label_src = pt_label_src.cpu().numpy()
        if has_label:
            nt_label = trace_m.flatten(3).argmax(-1)
            nt_label_tgt = torch.div(nt_label, self.nt_num_nodes, rounding_mode="floor")
            nt_label_src = torch.remainder(nt_label, self.nt_num_nodes)
            nt_label_tgt = nt_label_tgt.cpu().numpy()
            nt_label_src = nt_label_src.cpu().numpy()
        else:
            _naive_label = type("_NL", (object,), {"__getitem__": lambda *x: None})()
            nt_label_tgt = nt_label_src = _naive_label
        spans_ = []
        for i, spans_item in enumerate(spans):
            labels_spans_item = []
            for span in spans_item:
                if span[1] - span[0] == 1:
                    labels_spans_item.append(
                        (
                            span[0],
                            span[1],
                            "p",
                            pt_label_tgt[i, span[0]],
                            pt_label_src[i, span[0]],
                        )
                    )
                else:
                    labels_spans_item.append(
                        (
                            span[0],
                            span[1],
                            "n",
                            nt_label_tgt[i, span[0], span[1]],
                            nt_label_src[i, span[0], span[1]],
                        )
                    )
            spans_.append(labels_spans_item)
        return spans_

    def mbr_decoding_ngram_pt(self):
        # term: b, n, max_width, tgt_pt, src_pt
        # trace: b, n+1, n+1
        batch, seq_len = self.params["term"].shape[:2]
        marginal = self.marginal
        trace_m: Tensor = marginal["trace"]  # b n+1 n+1 tgtnt srcnt
        term_m: Tensor = marginal["term"].flatten(3)
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            spans = _cky_zero_order_ngram(term_m.sum(3), trace_m, self.lens)
        else:
            spans = [[(0, 1, "n"), (1, 2, "n"), (0, 2, "p")] for _ in range(batch)]
        pt_label = term_m.argmax(-1)
        pt_label_tgt = torch.div(pt_label, self.pt_num_nodes, rounding_mode="floor")
        pt_label_src = torch.remainder(pt_label, self.pt_num_nodes)
        pt_label_tgt = pt_label_tgt.cpu().numpy()
        pt_label_src = pt_label_src.cpu().numpy()
        spans_ = []
        for i, spans_item in enumerate(spans):
            labels_spans_item = []
            for span in spans_item:
                if span[2] == "p":
                    labels_spans_item.append(
                        (
                            span[0],
                            span[1],
                            "p",
                            pt_label_tgt[i, span[0], span[1] - span[0] - 1],
                            pt_label_src[i, span[0], span[1] - span[0] - 1],
                        )
                    )
                else:
                    labels_spans_item.append((span[0], span[1], "n", None, None))
            spans_.append(labels_spans_item)
        return spans_

    def is_log_param(self, name):
        i = self.KEYS.index(name)
        return self.LOGSPACE[i]

    @lazy_property
    def log_params(self):
        output = {}
        for k, is_in_log_space in zip(self.KEYS, self.LOGSPACE):
            pk = self.params[k]
            if not is_in_log_space:
                pk = (pk + 1e-9).log()
            output[k] = pk
        return output

    def spawn(self, **kwargs):
        # generate new decomp obj with some new args. this will call __init__
        # this assume all attributes are not renamed from __init__'s args
        for cls in type(self).__mro__:
            args = inspect.getfullargspec(cls.__init__)
            for argname in chain(args.args[1:], args.kwonlyargs):
                if (value := self.__dict__.get(argname)) is not None:
                    if argname in kwargs:
                        if argname == "params":
                            params = {**value, **kwargs.pop("params")}
                    else:
                        kwargs[argname] = value
            if args.varargs is None and args.varkw is None:
                break
        obj = object.__new__(type(self))
        obj.__init__(params, **kwargs)
        return obj


class DecompSamplerBase:
    def __init__(
        self,
        params: Dict[str, Tensor],
        nt_states,
        nt_num_nodes,
        pt_states,
        pt_num_nodes,
        batch_size,
        use_copy,
        num_samples=10,
        max_length=20,
        max_actions=100,
        strict=False,
        unk=1,
    ):
        super().__init__()
        self.nt_states = nt_states
        self.nt_num_nodes = nt_num_nodes
        self.pt_states = pt_states
        self.pt_num_nodes = pt_num_nodes
        self.batch_size = batch_size
        self.use_copy = use_copy
        self.num_samples = num_samples
        self.max_length = max_length
        self.max_actions = max_actions
        self.strict = strict
        self.unk = unk

        self.threshold = torch.nn.Threshold(1e-8, 0, True)
        self.params = self.process_params(params)

    def process_params(self, params):
        raise NotImplementedError

    def sample_impl(self, *args, **kwargs):
        raise NotImplementedError

    def get_params(self, bidx):
        return [p[bidx] for p in self.params]

    def get_kwargs(self):
        return {
            "nt_states": self.nt_states,
            "nt_num_nodes": self.nt_num_nodes,
            "pt_states": self.pt_states,
            "pt_num_nodes": self.pt_num_nodes,
            "use_copy": self.use_copy,
            "num_samples": self.num_samples,
            "max_length": self.max_length,
            "max_actions": self.max_actions,
            "unk": self.unk,
        }

    def __call__(self):
        preds = []
        for b in range(self.batch_size):
            samples, types, status = self.sample_impl(*self.get_params(b), **self.get_kwargs())
            half = len(status) // 2
            if (cnt := sum(item == _REACHLIMIT for item in status)) > half:
                log.warning(f"{cnt} trials are terminated due to REACHLIMIT")
            if (cnt := sum(item == _SONMASK for item in status)) > half:
                log.warning(f"{cnt} trials are terminated due to SONMASK")
            samples = [
                (sample, type_)
                for sample, type_, status_ in zip(samples, types, status)
                if len(sample) > 1 and (not self.strict or status_ == _OK)
            ]  # len=0 when max_actions is reached but no PT rules applied
            if len(samples) == 0:
                log.warning("All trials are failed.")
                samples = [([0, 0], [TokenType.VOCAB, TokenType.VOCAB])]
            preds.append(samples)
        return preds


@torch.no_grad()
def _cky_zero_order(marginals, lens):
    N = marginals.shape[-1]
    s = marginals.new_full(marginals.shape, -1e9)
    p = marginals.new_zeros(*marginals.shape, dtype=torch.long)
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

    p = p.cpu().numpy()
    lens = lens.cpu().numpy()
    spans = [backtrack(p[i], 0, length) for i, length in enumerate(lens)]
    for spans_item in spans:
        spans_item.sort(key=lambda i: i[1] - i[0])
    return spans


def backtrack(p, i, j):
    if j == i + 1:
        return [(i, j)]
    split = p[i][j]
    ltree = backtrack(p, i, split)
    rtree = backtrack(p, split, j)
    return [(i, j)] + ltree + rtree


@torch.no_grad()
def _cky_zero_order_ngram(pt_marginals, marginals, lens):
    # term: b, n, max_width
    # trace: b, n+1, n+1
    N = marginals.shape[-1]
    s = marginals.new_full(marginals.shape, -1e9)
    for w in range(min(pt_marginals.shape[2], N)):
        n = N - w - 1
        diagonal_copy_(s, pt_marginals[:, :n, w], w + 1)
    p = marginals.new_zeros(*marginals.shape, dtype=torch.long)
    pt_ind = marginals.new_zeros(*marginals.shape, dtype=torch.bool)
    for w in range(2, N):
        n = N - w
        starts = p.new_tensor(range(n))
        Y = stripe(s, n, w - 1, (0, 1))
        Z = stripe(s, n, w - 1, (1, w), 0)
        X, split = (Y + Z).max(2)
        x = X + diagonal(marginals, w)
        current = diagonal(s, w)
        is_pt = current > x
        x = torch.where(is_pt, current, x)
        diagonal_copy_(s, x, w)
        diagonal_copy_(p, split + starts.unsqueeze(0) + 1, w)
        diagonal_copy_(pt_ind, is_pt, w)

    p = p.cpu().numpy()
    pt_ind = pt_ind.cpu().numpy()
    lens = lens.cpu().numpy()
    spans = [backtrack_pt_ind(p[i], pt_ind[i], 0, length) for i, length in enumerate(lens)]
    for spans_item in spans:
        spans_item.sort(key=lambda i: i[1] - i[0])
    return spans


def backtrack_pt_ind(p, pt_ind, i, j):
    if j == i + 1 or pt_ind[i][j]:
        return [(i, j, "p")]
    split = p[i][j]
    ltree = backtrack_pt_ind(p, pt_ind, i, split)
    rtree = backtrack_pt_ind(p, pt_ind, split, j)
    return [(i, j, "n")] + ltree + rtree
