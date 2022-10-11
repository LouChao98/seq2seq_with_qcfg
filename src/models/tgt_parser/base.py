import dataclasses
from copy import copy
from dataclasses import dataclass
from functools import partial
from tkinter.messagebox import NO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence

from src.utils.fn import apply_to_nested_tensor

from ...datamodules.components.vocab import VocabularyPair
from ..constraint.base import RuleConstraintBase
from .struct.decomp_base import DecompBase, DecompSamplerBase, TokenType

# from operator import itemgetter


def itemgetter(list_of_indices):
    def _itemgetter(item):
        if isinstance(item, (torch.Tensor, np.ndarray)):
            return item[list_of_indices]
        elif isinstance(item, (list, tuple)):
            return [item[i] for i in list_of_indices]
        raise ValueError

    return _itemgetter


@dataclass
class TgtParserPrediction:
    batch_size: int
    nt: int
    nt_states: int
    nt_nodes: List[List[Tuple[int, int, int]]]
    nt_num_nodes: int
    pt: int
    pt_states: int
    pt_nodes: List[List[Tuple[int, int, int]]]
    pt_num_nodes: int
    params: Dict[str, torch.Tensor]
    device: torch.device

    # parsing
    posterior_params: Optional[Dict[str, torch.Tensor]] = None
    lengths: Optional[List[int]] = None
    dist: Optional[DecompBase] = None

    # sampling
    src: Optional[List[List[str]]] = None
    src_ids: Optional[torch.Tensor] = None
    sampler: Optional[DecompSamplerBase] = None

    def clear(self):
        self.posterior_params = None
        self.lengths = None
        self.dist = None
        self.src = None
        self.src_ids = None
        self.sampler = None
        return self

    def common(self):
        return {
            "nt_states": self.nt_states,
            "nt_num_nodes": self.nt_num_nodes,
            "pt_states": self.pt_states,
            "pt_num_nodes": self.pt_num_nodes,
            "batch_size": self.batch_size,
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            key = [key]
        getter = itemgetter(key)
        obj = copy(self)
        obj.params = apply_to_nested_tensor(obj.params, getter)
        if obj.posterior_params is not None:
            obj.posterior_params = apply_to_nested_tensor(obj.posterior_params, getter)
            obj.lengths = getter(obj.lengths)
            obj.dist = obj.dist.spawn(params=obj.posterior_params, lens=obj.lengths, **obj.common())
        if obj.src is not None:
            obj.src = getter(obj.src)
            obj.src_ids = getter(obj.src_ids)
            obj.sampler = None  # TODO method for rebuild sampler
        obj.batch_size = len(key)
        return obj

    def expand(self, size):
        assert self.batch_size == 1
        func = lambda x: x.expand(size, *[-1] * (x.ndim - 1))
        obj = copy(self)
        params = apply_to_nested_tensor(self.params, func)
        obj.params = params
        obj.batch_size = size
        if obj.posterior_params is not None:
            obj.posterior_params = apply_to_nested_tensor(self.posterior_params, func)
            assert isinstance(obj.lengths, list)
            obj.lengths = obj.lengths * size
            obj.dist = type(obj.dist)(obj.posterior_params, obj.lengths, **self.common())
        if obj.src is not None:
            obj.src = obj.src * size
            obj.src_ids = obj.src_ids.expand(size, -1)
            obj.sampler = None  # TODO method for rebuild sampler
        return obj

    def get_and_expand(self, k, size):
        # get k-th sample from batch and expand to size
        func = lambda x: x[k, None].expand(size, *[-1] * (x.ndim - 1))
        obj = copy(self)
        params = apply_to_nested_tensor(self.params, func)
        obj.params = params
        obj.batch_size = size
        if obj.posterior_params is not None:
            obj.posterior_params = apply_to_nested_tensor(self.posterior_params, func)
            assert isinstance(obj.lengths, list)
            obj.lengths = [obj.lengths[k]] * size
            obj.dist = type(obj.dist)(obj.posterior_params, obj.lengths, **self.common())
        if obj.src is not None:
            obj.src = [obj.src[k] * size]
            obj.src_ids = obj.src_ids[k, None].expand(size, -1)
            obj.sampler = None  # TODO method for rebuild sampler
        return obj


@dataclass
class DirTgtParserPrediction(TgtParserPrediction):
    direction: int = 0

    def common(self):
        return {**super().common(), "direction": self.direction}


class TgtParserBase(nn.Module):
    # spans should have inclusive boundries.

    def __init__(
        self,
        pt_states: int = 10,
        nt_states: int = 10,
        pt_span_range: Tuple = (1, 1),
        nt_span_range: Tuple = (2, 1000),
        use_copy: bool = False,
        vocab_pair: Optional[VocabularyPair] = None,
        rule_hard_constraint=None,
        rule_soft_constraint=None,
        rule_soft_constraint_solver=None,
        generation_max_length: int = 40,
        generation_max_actions: int = 80,
        generation_num_samples: int = 10,
        generation_ppl_batch_size: int = 1,
        generation_ppl_strict: bool = False,
    ):
        super().__init__()

        self.pt_states = pt_states
        self.nt_states = nt_states
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.use_copy = use_copy
        self.vocab_pair = vocab_pair

        self.rule_hard_constraint: Optional[RuleConstraintBase] = instantiate(rule_hard_constraint)
        self.rule_soft_constraint: Optional[RuleConstraintBase] = instantiate(rule_soft_constraint)
        self.rule_soft_constraint_solver = instantiate(rule_soft_constraint_solver)

        assert (
            self.rule_soft_constraint is None or self.rule_soft_constraint_solver is not None
        ), "A solver is required."

        self.generation_max_length = generation_max_length
        self.generation_max_actions = generation_max_actions
        self.generation_num_samples = generation_num_samples
        self.generation_ppl_batch_size = generation_ppl_batch_size
        self.generation_ppl_strict = generation_ppl_strict

        self.neg_huge = -1e9

    def forward(self, node_features, spans, **kwargs) -> TgtParserPrediction:
        raise NotImplementedError

    def observe_x(self, pred: TgtParserPrediction, x, lengths, inplace=True, **kwargs) -> TgtParserPrediction:
        if not inplace:
            pred = copy(pred).clear()
        pred.posterior_params = pred.params | self.build_rules_give_tgt(
            x,
            pred.params["term"],
            pred.params["root"],
            pred.pt_num_nodes,
            pred.pt_nodes,
            pred.nt_num_nodes,
            pred.nt_nodes,
            pred.pt,
            pred.nt,
            **kwargs,
        )
        pred.lengths = lengths
        return pred

    def prepare_sampler(self, pred: TgtParserPrediction, src, src_ids, inplace=True) -> TgtParserPrediction:
        if not inplace:
            pred = copy(pred).clear()
        pred.src = src
        pred.src_ids = src_ids
        return pred

    def get_soft_constraint_loss(self, pred):
        if self.rule_soft_constraint_solver is None:
            return 0

        constraint_feature = self.rule_soft_constraint.get_feature_from_pred(pred)
        return self.rule_soft_constraint_solver(pred, constraint_feature)

    def parse(self, pred: TgtParserPrediction):
        assert pred.dist is not None
        out = pred.dist.mbr_decoded
        # find alignments
        aligned_spans = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pred.pt_nodes, pred.nt_nodes)):
            aligned_spans_item = []
            for l, r, label in all_span:
                # try:
                if l == r:
                    aligned_spans_item.append(pt_span[label % pred.pt_num_nodes])
                else:
                    aligned_spans_item.append(nt_span[label % pred.nt_num_nodes])
            # except IndexError:
            #     breakpoint()
            aligned_spans.append(aligned_spans_item)
        return out, aligned_spans, pred.pt_nodes, pred.nt_nodes

    def generate(self, pred: TgtParserPrediction):
        assert pred.sampler is not None
        preds = pred.sampler()

        if self.use_copy:
            preds, copy_positions = self.expand_preds_using_copy(
                pred.src_ids,
                preds,
                pred.pt_nodes,
                pred.nt_nodes,
                self.generation_max_length,
                pred.device,
            )
        else:
            preds, copy_positions = self.expand_preds_not_using_copy(preds)

        preds = self.choose_samples_by_ppl(preds, copy_positions, pred)
        preds = self.to_str_tokens(preds, pred.src)
        return preds

    def expand_preds_using_copy(self, src_ids, preds, pt_spans, nt_spans, max_len, device):
        # expand copied spans and build copy_position
        vocab_pair = self.vocab_pair
        src_lens = (src_ids != vocab_pair.src.pad_token_id).sum(1).tolist()
        src_ids = src_ids.tolist()
        preds_ = []
        copy_positions = []

        for batch, pt_spans_item, nt_spans_item, src_ids_item, src_len in zip(
            preds, pt_spans, nt_spans, src_ids, src_lens
        ):
            expanded_batch = []
            copy_pts = []
            copy_nts = []
            copy_unks = []
            for item in batch:
                expanded = []
                copy_pt = np.zeros((src_len, max_len), dtype=np.bool8)
                copy_nt = [[] for _ in range(max_len)]  # no need to prune
                copy_unk = {}  # record position if copy unk token
                for v, t in zip(item[0], item[1]):
                    if t == TokenType.VOCAB:
                        if len(expanded) + 1 > max_len:
                            break
                        expanded.append(v)
                    elif t == TokenType.COPY_PT:
                        span = pt_spans_item[v]
                        tokens = vocab_pair.src2tgt(src_ids_item[span[0] : span[1] + 1])
                        if len(expanded) + len(tokens) > max_len:
                            break
                        copy_pt[span[0], len(expanded)] = True
                        if tokens[0] == vocab_pair.tgt.unk_token_id:
                            copy_unk[len(expanded)] = span[0]
                        expanded.extend(tokens)
                    elif t == TokenType.COPY_NT:
                        span = nt_spans_item[v]
                        tokens = vocab_pair.src2tgt(src_ids_item[span[0] : span[1] + 1])
                        if len(expanded) + len(tokens) > max_len:
                            break
                        copy_nt[span[1] - span[0] - 1].append((span[0], len(expanded)))  # copy_nt starts from w=2
                        for i, token in enumerate(tokens):
                            if token == vocab_pair.tgt.unk_token_id:
                                copy_unk[len(expanded) + i] = span[0] + i
                        expanded.extend(tokens)

                if max(expanded) >= len(vocab_pair.tgt):
                    assert False, "This should never happen"
                if len(expanded) > max_len:
                    continue
                expanded_batch.append(expanded)
                copy_pts.append(copy_pt)
                copy_nts.append(copy_nt)
                copy_unks.append(copy_unk)
            copy_pts = torch.from_numpy(np.stack(copy_pts, axis=0)).to(device)
            copy_positions.append((copy_pts, copy_nts, copy_unks))
            preds_.append(expanded_batch)
        return preds_, copy_positions

    def expand_preds_not_using_copy(self, preds):
        preds_ = []
        copy_positions = []
        for batch in preds:
            expanded_batch = []
            for item in batch:
                expanded_batch.append(item[0])
            copy_positions.append((None, None, None))
            preds_.append(expanded_batch)
        return preds_, copy_positions

    @torch.no_grad()
    def choose_samples_by_ppl(self, preds, copy_positions, pred: TgtParserPrediction):
        padid = self.vocab_pair.tgt.pad_token_id or 0
        new_preds = []
        pred = copy(pred).clear()
        for bidx, (
            preds_item,
            (copy_pt, copy_nt, copy_unk),
        ) in enumerate(zip(preds, copy_positions)):
            sort_id = list(range(len(preds_item)))
            sort_id.sort(key=lambda i: len(preds_item[i]), reverse=True)
            _ids = [torch.tensor(preds_item[i]) for i in sort_id]
            _lens = [len(item) for item in _ids]
            _ids_t = pad_sequence(_ids, batch_first=True, padding_value=padid)
            _ids_t = _ids_t.to(pred.device)

            if copy_pt is not None:
                copy_pt = copy_pt[sort_id]
                copy_nt = [copy_nt[i] for i in sort_id]
                copy_unk = [copy_unk[i] for i in sort_id]

            batch_size = pred.batch_size if self.generation_ppl_batch_size is None else self.generation_ppl_batch_size

            ppl = []
            for j in range(0, len(_ids), batch_size):
                if copy_pt is not None:
                    _pt_copy = copy_pt[j : j + batch_size, :, : _ids_t.shape[1]]
                    _nt_copy = copy_nt[j : j + batch_size]
                else:
                    _pt_copy, _nt_copy = None, None

                batch_t = _ids_t[j : j + batch_size]
                batch_l = _lens[j : j + batch_size]
                sub_pred = pred[bidx].expand(len(batch_t))
                sub_pred = self.observe_x(sub_pred, batch_t, batch_l, pt_copy=_pt_copy, nt_copy=_nt_copy)
                nll = sub_pred.dist.nll.detach().cpu().numpy()
                ppl.append(np.exp(nll / np.array(_lens[j : j + batch_size])))

            ppl = np.concatenate(ppl, 0)
            assert not np.any(np.isnan(ppl))
            chosen = np.argmin(ppl)
            new_preds.append(
                (
                    _ids[chosen],
                    ppl[chosen],
                    None if copy_unk is None else copy_unk[chosen],
                )
            )
        return new_preds

    def to_str_tokens(self, preds, src):
        tgt_vocab = self.vocab_pair.tgt
        pred_strings = []
        for pred, src_sent in zip(preds, src):
            snt, score, copy_unk = pred
            try:
                sent = tgt_vocab.convert_ids_to_tokens(snt)
                if copy_unk is not None:
                    for t, s in copy_unk.items():
                        sent[t] = src_sent[s]
                pred_strings.append((sent, score))
            except IndexError:
                print("Bad pred:", snt)
                pred_strings.append([("", -999)])
        return pred_strings

    def build_src_features(self, spans, node_features):
        # seperate nt and pt features according to span width
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for spans_item, node_features_item in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for s, f in zip(spans_item, node_features_item):
                s_len = s[1] - s[0] + 1
                if self.nt_span_range[0] <= s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(f)
                    nt_span.append(s)
                if self.pt_span_range[0] <= s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(f)
                    pt_span.append(s)
            self.sanity_check_spans(nt_span, pt_span)
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_num_nodes_list = [len(item) for item in nt_node_features]
        pt_num_nodes_list = [len(item) for item in pt_node_features]
        nt_node_features = pad_sequence(nt_node_features, batch_first=True, padding_value=0.0)
        pt_node_features = pad_sequence(pt_node_features, batch_first=True, padding_value=0.0)
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
        return (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
        )

    @staticmethod
    def sanity_check_spans(nt_spans, pt_spans):
        num_terms = sum(item[0] == item[1] for item in pt_spans)
        # there must be something in nt_spans
        assert len(nt_spans) > 0
        # root must be the last of nt_spans
        assert nt_spans[-1][0] == 0 and nt_spans[-1][1] == num_terms - 1
        # singles must be ordered and placed at the begining of pt_spans
        for i, item in zip(range(num_terms), pt_spans):
            assert item[0] == item[1] == i

    def build_rules_give_tgt(
        self,
        tgt: torch.Tensor,
        term: torch.Tensor,
        root: torch.Tensor,
        max_pt_spans: int,
        pt_spans: List[List[Tuple[int, int, int]]],
        max_nt_spans: int,
        nt_spans: List[List[Tuple[int, int, int]]],
        pt: int,
        nt: int,
        pt_copy=None,
        nt_copy=None,
        observed_mask=None,
    ):
        batch_size, n = tgt.shape[:2]
        term = term.unsqueeze(1).expand(batch_size, n, pt, term.size(2))
        tgt_expand = tgt.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        term = torch.gather(term, 3, tgt_expand).squeeze(3)

        constraint_scores = None
        lse_scores = None
        add_scores = None

        if self.use_copy:
            if pt_copy is not None:
                term = self.build_pt_copy_constraint(term, pt_copy)
            if nt_copy is not None:
                constraint_scores = self.build_nt_copy_constraint(
                    batch_size,
                    n,
                    max_nt_spans,
                    nt_spans,
                    nt_copy,
                    constraint_scores,
                )
            else:
                constraint_scores = self.get_init_nt_constraint(batch_size, n, max_nt_spans)

            root = root.clone().view(batch_size, self.nt_states, -1)
            root[:, -1] = self.neg_huge
            root = root.view(batch_size, -1)

        if constraint_scores is not None:
            constraint_scores = self.post_process_nt_constraint(constraint_scores, tgt.device)

        if observed_mask is not None:
            constraint_scores = self.build_observed_span_constraint(
                batch_size, n, max_nt_spans, observed_mask, constraint_scores
            )

        return {"term": term, "root": root, "constraint": constraint_scores, "lse": lse_scores, "add": add_scores}

    def build_pt_copy_constraint(self, terms, constraint):
        batch_size, n = terms.shape[:2]
        terms = terms.view(batch_size, n, self.pt_states, -1)
        copy_m = constraint[:, : terms.shape[-1]].transpose(1, 2)
        terms[:, :, -1, : copy_m.shape[2]] = self.neg_huge * ~copy_m
        terms = terms.view(batch_size, n, -1)
        return terms

    def get_init_nt_constraint(self, batch_size, n, max_nt_spans):
        return [
            (
                np.full(
                    (batch_size, n - w, self.nt_states, max_nt_spans),
                    self.neg_huge,
                    dtype=np.float32,
                ),
                np.zeros((batch_size, n - w, self.nt_states, max_nt_spans), dtype=np.bool8),
            )
            for w in range(1, n)
        ]

    def build_nt_copy_constraint(self, batch_size, n, max_nt_spans, nt_spans, copy_position, constraint=None):
        # n: max_length
        if constraint is None:
            constraint = self.get_init_nt_constraint(batch_size, n, max_nt_spans)
        for batch_idx, (nt_spans_inst, possible_copy) in enumerate(zip(nt_spans, copy_position)):
            for i, (l, r, _) in enumerate(nt_spans_inst):
                w = r - l - 1
                t = None
                if w >= len(possible_copy) or w < 0:
                    continue
                for possible_s, possible_t in possible_copy[w]:
                    if possible_s == l:
                        t = possible_t
                        break
                if t is not None:
                    constraint[w][0][batch_idx, t, -1, i] = 0
        for value, mask in constraint:
            mask[:, :, -1] = True
        return constraint

    def build_observed_span_constraint(self, batch_size, n, max_nt_spans, observed_constraint, constraint=None):
        if constraint is None:
            constraint = self.get_init_nt_constraint(batch_size, n, max_nt_spans)
        for item, (value, mask) in zip(observed_constraint, constraint):
            mask |= item.view(list(item.shape) + [1] * (mask.ndim - item.ndim))
            value[item] = self.neg_huge
        return constraint

    def post_process_nt_constraint(self, constraint, device):
        constraint_ = []
        for value, mask in constraint:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value.reshape(value.shape[:2] + (-1,)))
                mask = torch.from_numpy(mask.reshape(value.shape))
            else:
                value = value.flatten(2)
                mask = mask.flatten(2)
            constraint_.append((value.to(device), mask.to(device)))
        return constraint_

    def merge_nt_constraint(self, constraint1, constraint2):
        # write constraint2 to constraint1. all should be postprocessed.
        # mask2=False  =>  value=value1
        # mask2=True   =>  value=value2
        # mask = mask1 | mask2

        merged = []
        for (v1, m1), (v2, m2) in zip(constraint1, constraint2):
            v = torch.where(m2, v2, v1)
            m = m1 | m2
            merged.append((v, m))
        return merged

    def sampler_common(self):
        return {
            "use_copy": self.use_copy,
            "num_samples": self.generation_num_samples,
            "max_length": self.generation_max_length,
            "max_actions": self.generation_max_actions,
            "strict": self.generation_ppl_strict,
            "unk": 1,
        }
