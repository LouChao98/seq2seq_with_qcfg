from logging import root
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import VocabularyPair
from .struct.pcfg import PCFG, TokenType


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
        generation_num_samples: int = 10,
        generation_ppl_batch_size: int = 1,
    ):
        super().__init__()

        self.pcfg = None
        self.pt_states = pt_states
        self.nt_states = nt_states
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.use_copy = use_copy
        self.vocab_pair = vocab_pair

        self.rule_hard_constraint = instantiate(rule_hard_constraint)
        self.rule_soft_constraint = instantiate(rule_soft_constraint)
        self.rule_soft_constraint_solver = instantiate(
            rule_soft_constraint_solver, pcfg=PCFGProxy(self, "pcfg")
        )
        self._pcfg = PCFGProxy(self, "pcfg")
        assert (
            self.rule_soft_constraint is None
            or self.rule_soft_constraint_solver is not None
        ), "A solver is required."

        self.generation_max_length = generation_max_length
        self.generation_num_samples = generation_num_samples
        self.generation_ppl_batch_size = generation_ppl_batch_size

        self.neg_huge = -1e9

    def forward(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params, *_ = self.get_params(node_features, spans, x, **kwargs)

        out = self.pcfg(params, lengths)
        return out

    def get_soft_constraint_loss(
        self, x, lengths, node_features, spans, params=None, **kwargs
    ):
        if self.rule_soft_constraint_solver is None:
            return 0
        if params is None:
            params = self.get_params(node_features, spans, x, **kwargs)
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = params
        batch_size = len(lengths)
        device = x.device
        constraint_feature = self.rule_soft_constraint.get_feature(
            batch_size, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
        )
        return self.rule_soft_constraint_solver(
            params,
            lengths,
            constraint_feature,
        )

    def parse(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params = self.get_params(node_features, spans, x, **kwargs)
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = params

        out = self.pcfg(params, lengths, decode=True)

        # find alignments
        aligned_spans = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            aligned_spans_item = []
            for l, r, label in all_span:
                if l == r:
                    aligned_spans_item.append(pt_span[label % pt_num_nodes])
                else:
                    aligned_spans_item.append(nt_span[label % nt_num_nodes])
            aligned_spans.append(aligned_spans_item)
        return out, aligned_spans, pt_spans, nt_spans

    def generate(
        self,
        node_features,
        spans,
        src_ids: torch.Tensor,
        src: List[List[str]],
        params=None,
        **kwargs,
    ):
        if params is None:
            params = self.get_params(node_features, spans, **kwargs)
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = params

        preds = self.pcfg.sampled_decoding(
            params,
            nt_spans,
            self.nt_states,
            pt_spans,
            self.pt_states,
            use_copy=self.use_copy,
            num_samples=self.generation_num_samples,
            max_length=self.generation_max_length,
        )

        if self.use_copy:
            preds, copy_positions = self.expand_preds_using_copy(
                src_ids,
                preds,
                pt_spans,
                nt_spans,
                self.generation_max_length,
                node_features[0].device,
            )
        else:
            preds, copy_positions = self.expand_preds_not_using_copy(preds)

        preds = self.choose_samples_by_ppl(preds, copy_positions, spans, node_features)
        preds = self.to_str_tokens(preds, src)
        return preds

    def expand_preds_using_copy(
        self, src_ids, preds, pt_spans, nt_spans, max_len, device
    ):
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
                        copy_nt[span[1] - span[0] - 1].append(
                            (span[0], len(expanded))
                        )  # copy_nt starts from w=2
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
        return preds, copy_positions

    @torch.no_grad()
    def choose_samples_by_ppl(self, preds, copy_positions, src_spans, node_features):
        padid = self.vocab_pair.tgt.pad_token_id or 0
        new_preds = []
        for (
            preds_item,
            (copy_pt, copy_nt, copy_unk),
            src_spans_item,
            node_features_item,
        ) in zip(preds, copy_positions, src_spans, node_features):
            sort_id = list(range(len(preds_item)))
            sort_id.sort(key=lambda i: len(preds_item[i]), reverse=True)
            _ids = [torch.tensor(preds_item[i]) for i in sort_id]
            _lens = [len(item) for item in _ids]
            _ids_t = pad_sequence(_ids, batch_first=True, padding_value=padid)
            _ids_t = _ids_t.to(node_features[0].device)

            if copy_pt is not None:
                copy_pt = copy_pt[sort_id]
                copy_nt = [copy_nt[i] for i in sort_id]
                copy_unk = [copy_unk[i] for i in sort_id]

            batch_size = (
                len(node_features)
                if self.generation_ppl_batch_size is None
                else self.generation_ppl_batch_size
            )

            ppl = []
            for j in range(0, len(_ids), batch_size):
                real_batch_size = min(batch_size, len(_ids) - j)
                _node_ft = [node_features_item for _ in range(real_batch_size)]
                _spans = [src_spans_item for _ in range(real_batch_size)]
                if copy_pt is not None:
                    _copy = (
                        copy_pt[j : j + batch_size, :, : _ids_t.shape[1]],
                        copy_nt[j : j + batch_size],
                    )
                else:
                    _copy = None
                nll = (
                    self(
                        _ids_t[j : j + batch_size],
                        _lens[j : j + batch_size],
                        _node_ft,
                        _spans,
                        copy_position=_copy,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
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

    def get_params(
        self, node_features, spans, x: Optional[torch.Tensor] = None, **kwargs
    ):
        raise NotImplementedError

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
        nt_node_features = pad_sequence(
            nt_node_features, batch_first=True, padding_value=0.0
        )
        pt_node_features = pad_sequence(
            pt_node_features, batch_first=True, padding_value=0.0
        )
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
        terms: torch.Tensor,
        roots: torch.Tensor,
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
        terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
        tgt_expand = tgt.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        terms = torch.gather(terms, 3, tgt_expand).squeeze(3)

        constraint_scores = None
        lse_scores = None
        add_scores = None

        if self.use_copy:
            if pt_copy is not None:
                terms = self.build_pt_copy_constraint(terms, pt_copy)
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
                constraint_scores = self.get_init_nt_constraint(
                    batch_size, n, max_nt_spans
                )

            roots = roots.clone().view(batch_size, self.nt_states, -1)
            roots[:, -1] = self.neg_huge
            roots = roots.view(batch_size, -1)

        if observed_mask is not None:
            constraint_scores = self.build_observed_span_constraint(
                batch_size, n, max_nt_spans, observed_mask, constraint_scores
            )

        if constraint_scores is not None:
            constraint_scores = self.post_process_nt_constraint(
                constraint_scores, tgt.device
            )

        return terms, roots, constraint_scores, lse_scores, add_scores

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
                np.zeros(
                    (batch_size, n - w, self.nt_states, max_nt_spans), dtype=np.bool8
                ),
            )
            for w in range(1, n)
        ]

    def build_nt_copy_constraint(
        self, batch_size, n, max_nt_spans, nt_spans, copy_position, constraint=None
    ):
        # n: max_length
        if constraint is None:
            constraint = self.get_init_nt_constraint(batch_size, n, max_nt_spans)
        for batch_idx, (nt_spans_inst, possible_copy) in enumerate(
            zip(nt_spans, copy_position)
        ):
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

    def build_observed_span_constraint(
        self, batch_size, n, max_nt_spans, observed_constraint, constraint=None
    ):
        if constraint is None:
            constraint = self.get_init_nt_constraint(batch_size, n, max_nt_spans)
        for item, (value, mask) in zip(observed_constraint, constraint):
            mask |= item[..., None, None]
        return constraint

    def post_process_nt_constraint(self, constraint, device):
        constraint_ = []
        for value, mask in constraint:
            value = torch.from_numpy(value.reshape(value.shape[:2] + (-1,)))
            mask = torch.from_numpy(mask.reshape(value.shape))
            constraint_.append((value.to(device), mask.to(device)))
        return constraint_


class PCFGProxy:
    def __init__(self, obj, attribute):
        self.obj = obj
        self.attribute = attribute
        self.attribute_obj = None

    def __getattr__(self, key):
        if (attribute_obj := self.__dict__["attribute_obj"]) is None:
            obj = self.__dict__["obj"]
            attribute = self.__dict__["attribute"]
            attribute_obj = getattr(obj, attribute)
            self.__dict__["attribute_obj"] = attribute_obj
        return getattr(attribute_obj, key)

    def __call__(self, *args, **kwargs):
        if (attribute_obj := self.__dict__["attribute_obj"]) is None:
            obj = self.__dict__["obj"]
            attribute = self.__dict__["attribute"]
            attribute_obj = getattr(obj, attribute)
            self.__dict__["attribute_obj"] = attribute_obj
        return attribute_obj(*args, **kwargs)
