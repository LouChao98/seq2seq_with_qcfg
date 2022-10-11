import inspect
import logging
from enum import IntEnum
from itertools import chain
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.autograd import grad
from torch.distributions.utils import lazy_property

from ._fn import diagonal, diagonal_copy_, stripe

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
        assert (lens[1:] <= lens[:-1]).all(), "Expect lengths in descending."

        self._traced_cache = None
        self._untraced_cache = None

    def __call__(self, trace=None):
        if trace is None:
            trace = not self.no_trace and torch.is_grad_enabled()
        if self._traced_cache is not None:
            return self._traced_cache
        if not trace and self._untraced_cache is not None:
            return self._untraced_cache
        result = self.inside_handler(trace)
        if trace:
            self._traced_cache = result
        else:
            self._untraced_cache = result
        return result

    def inside_handler(self, trace):
        if trace:
            with torch.enable_grad():
                return self.inside(trace)
        else:
            return self.inside(trace)

    def inside(self, trace):
        raise NotImplementedError

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

    @property
    def partition(self):
        return self()[0]

    @property
    def nll(self):
        return -self()[0]

    @lazy_property
    def _compute_marginal(self):
        logZ, trace = self()
        logZ.sum().backward()
        return trace

    @property
    def marginal(self):
        # logZ, trace = self()
        # return grad(logZ.sum(), [trace], create_graph=True, retain_graph=True)[0]
        # logZ.sum().backward()
        trace = self._compute_marginal
        return trace.grad

    @property
    def marginal_rule(self):
        _ = self._compute_marginal
        output = {}
        for k, is_in_log_space, m in zip(self.KEYS, self.LOGSPACE):
            g = self.params[k].grad
            if is_in_log_space:
                output[k] = g
            else:
                output[k] = m * self.params[k]
        return output
        # logZ, trace = self()
        # marginals = grad(
        #     logZ.sum(),
        #     [self.params[k] for k in self.KEYS],
        #     create_graph=True,
        #     retain_graph=True,
        #     allow_unused=False,
        # )
        # marginals_ = []
        # for k, is_in_log_space, m in zip(self.KEYS, self.LOGSPACE, marginals):
        #     if is_in_log_space:
        #         marginals_.append(m)
        #     else:
        #         marginals_.append(m * self.params[k])
        # return dict(zip(self.KEYS, marginals_))

    @lazy_property
    def entropy(self):
        marginal = self.marginal_rule
        result = self.partition.clone()
        for k in self.KEYS:
            result -= (self.log_params[k] * marginal[k]).flatten(1).sum(-1)
        return result

    def cross_entropy(self, other: "DecompBase", fix_left=False):
        # self = p, other = q, ce(q, p)
        q = self
        p = other
        qm = q.marginal_rule
        result = p.partition.clone()
        for k in self.KEYS:
            qmk = qm[k]
            if fix_left:
                qmk = qmk.detach()
            result -= (qmk * p.log_params[k]).flatten(1).sum(-1)
        return result

    def kl(self, other: "DecompBase", fix_left=False):
        q = self
        p = other
        qm = q.marginal_rule
        result = p.partition.clone() - q.partition.clone()
        for k in self.KEYS:
            qmk = qm[k]
            if fix_left:
                qmk = qmk.detach()
            result += (qmk * (q.log_params[k] - p.log_params[k])).flatten(1).sum(-1)
        return result

    @lazy_property
    def mbr_decoded(self):
        batch, seq_len = self.params["term"].shape[:2]
        marginals = self.marginal.detach()
        # to avoid some trivial corner cases.
        if seq_len >= 3:
            scores = marginals.flatten(3).sum(-1)
            spans = _cky_zero_order(scores, self.lens)
        else:
            # minimal length is 2
            spans = [[(0, 1), (1, 2), (0, 2)] for _ in range(batch)]
        spans_ = []
        need_fix = marginals.ndim == 5
        labels = marginals.flatten(3).argmax(-1).cpu().numpy()
        for i, spans_inst in enumerate(spans):
            labels_spans_inst = []
            for span in spans_inst:
                labels_spans_inst.append((span[0], span[1] - 1, labels[i, span[0], span[1]]))
            spans_.append(labels_spans_inst)
        if need_fix:
            spans_ = self.recompute_label_for_flex(spans_, self.pt_num_nodes, self.nt_num_nodes)
        return spans_

    @staticmethod
    def convert_to_tree(spans, length):
        tree = [(i, str(i)) for i in range(length)]
        tree = dict(tree)
        for l, r, _ in spans:
            if l != r:
                span = "({} {})".format(tree[l], tree[r])
                tree[r] = tree[l] = span
        return tree[0]

    @staticmethod
    def recompute_label_for_flex(spans, pt_spans, nt_spans):
        processed = []
        max_spans = max(pt_spans, nt_spans)
        for spans_inst in spans:
            newspans = []
            for left, right, label in spans_inst:
                symbol, alignment = divmod(label, max_spans)
                if left == right:
                    newspans.append((left, right, symbol * pt_spans + alignment))
                else:
                    newspans.append((left, right, symbol * nt_spans + alignment))
            processed.append(newspans)
        return processed

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
            if (cnt := sum(item == _REACHLIMIT for item in status)) > 0:
                log.warning(f"{cnt} trials are terminated due to REACHLIMIT")
            if (cnt := sum(item == _SONMASK for item in status)) > 0:
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
    for spans_inst in spans:
        spans_inst.sort(key=lambda i: i[1] - i[0])
    return spans


def backtrack(p, i, j):
    if j == i + 1:
        return [(i, j)]
    split = p[i][j]
    ltree = backtrack(p, i, split)
    rtree = backtrack(p, split, j)
    return [(i, j)] + ltree + rtree
