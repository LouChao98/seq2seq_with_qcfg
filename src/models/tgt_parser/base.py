import logging
import math
import random
from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.datamodules.datamodule import _DataModule
from src.utils.fn import apply_to_nested_tensor

from ..constraint.base import RuleConstraintBase
from ..struct.base import DecompBase, DecompSamplerBase, TokenType

logger = logging.getLogger(__file__)


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

    nt_features: torch.Tensor = None
    pt_features: torch.Tensor = None

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
        obj.pt_nodes = [obj.pt_nodes[k] for k in key]
        obj.nt_nodes = [obj.nt_nodes[k] for k in key]
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
        obj.pt_nodes *= size
        obj.nt_nodes *= size
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
        obj.pt_nodes = [obj.pt_nodes[k]] * size
        obj.nt_nodes = [obj.nt_nodes[k]] * size
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


NO_COPY_SPAN = -103


class TgtParserBase(nn.Module):
    # spans should have inclusive boundries.

    def __init__(
        self,
        pt_states: int = 10,
        nt_states: int = 10,
        pt_span_range: Tuple = (1, 1),
        nt_span_range: Tuple = (2, 1000),
        use_copy: bool = False,
        use_observed: bool = False,
        datamodule: Optional[_DataModule] = None,
        rule_hard_constraint=None,
        rule_soft_constraint=None,
        rule_soft_constraint_solver=None,
        rule_reweight_constraint=None,
        rule_reweight_test_only=False,
        generation_criteria: str = "ppl",
        generation_max_length: int = 40,
        generation_max_actions: int = 80,
        generation_num_samples: int = 10,
        generation_ppl_batch_size: int = 1,
        generation_strict: bool = False,
    ):
        super().__init__()

        self.pt_states = pt_states
        self.nt_states = nt_states
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.use_copy = use_copy
        self.use_observed = use_observed
        self.datamodule = datamodule

        self.rule_hard_constraint: Optional[RuleConstraintBase] = instantiate(rule_hard_constraint)
        self.rule_soft_constraint: Optional[RuleConstraintBase] = instantiate(rule_soft_constraint)
        self.rule_soft_constraint_solver = instantiate(rule_soft_constraint_solver)
        self.rule_reweight_constraint: Optional[RuleConstraintBase] = instantiate(rule_reweight_constraint)
        self.rule_reweight_test_only = rule_reweight_test_only

        assert generation_criteria in ("ppl", "likelihood", "contrastive", "none")

        self.generation_criteria = generation_criteria
        self.generation_max_length = generation_max_length
        self.generation_max_actions = generation_max_actions
        self.generation_num_samples = generation_num_samples
        self.generation_ppl_batch_size = generation_ppl_batch_size
        self.generation_strict = generation_strict

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

    def get_soft_constraint_loss(self, pred: TgtParserPrediction):
        constraint_feature = self.rule_soft_constraint.get_feature_from_pred(pred)
        return self.rule_soft_constraint_solver(pred, constraint_feature)

    def get_rl_loss(self, pred: TgtParserPrediction, ent_reg):
        assert ent_reg > 0
        # log pow(partition, 1/(len-1))
        partition = pred.dist.partition / (torch.tensor(pred.lengths, device=pred.device) - 1)
        reward = self.rule_soft_constraint.get_weight_from_pred(pred)
        reward = reward + partition[:, None, None, None] - np.log(ent_reg)
        dist = pred.dist.spawn(params={"rule": pred.dist.params["rule"] + reward})
        return dist.nll

    def get_raml_loss(self, pred: TgtParserPrediction):
        reward = self.rule_soft_constraint.get_weight_from_pred(pred)
        dist = pred.dist.spawn(
            params={
                "term": torch.where(pred.dist.params["term"] > -1e8, 0, -1e9),
                "rule": torch.where(pred.dist.params["rule"] > -1e8, reward, -1e9),
                "root": torch.where(pred.dist.params["root"] > -1e8, 0.0, -1e9),
            }
        )
        return dist.cross_entropy(pred.dist, fix_left=True) + pred.dist.nll

    def get_noisy_span_loss(self, node_features, node_spans, num_or_ratio, observes):
        noisy_features, noisy_spans = [], []
        bsz = len(node_features)
        for bidx in range(bsz):
            num = num_or_ratio if isinstance(num_or_ratio, int) else math.ceil(len(node_spans[bidx]) * num_or_ratio)
            features_item, spans_item = [], []
            for _ in range(num):
                _i = random.choice([i for i in range(bsz) if i != bidx])
                # do not add root and leaves
                _ii = random.choice([i for i, span in enumerate(node_spans[_i][:-1]) if span[1] - span[0] > 0])
                features_item.append(node_features[_i][_ii, None].detach())
                span = node_spans[_i][_ii]
                spans_item.append((span[0], span[1], NO_COPY_SPAN))
            noisy_features.append(torch.cat([node_features[bidx]] + features_item, dim=0))
            noisy_spans.append(node_spans[bidx][:-1] + spans_item + node_spans[bidx][-1:])
        pred = self(noisy_features, noisy_spans)
        noisy_nodes = [torch.tensor([span[2] == NO_COPY_SPAN for span in spans]) for spans in pred.nt_nodes]
        noisy_nodes = pad_sequence(noisy_nodes, batch_first=True, padding_value=False).to(pred.device)  # b srcnt
        impl = 2
        if impl == 1:
            pred = self.observe_x(pred, **observes)  # TODO handle hard constraint
            marginal = pred.dist.marginal_with_grad[1]  # b n n tgt_nt src_nt
            loss = marginal * noisy_nodes[:, None, None, None].expand_as(marginal)
            loss = loss.flatten(1).sum(1)
        else:
            rule = pred.params["rule"]
            shape = rule.shape
            rule = rule.view(pred.batch_size, pred.nt_states, pred.nt_num_nodes, -1)
            rule = torch.where(noisy_nodes[:, None, :, None].expand_as(rule), -1e9, rule)
            pred.params["rule"] = rule.view(shape)
            pred = self.observe_x(pred, **observes)
            loss = pred.dist.nll
        return loss

    def parse(self, pred: TgtParserPrediction):
        assert pred.dist is not None
        out = pred.dist.decoded

        # find alignments
        aligned_spans = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pred.pt_nodes, pred.nt_nodes)):
            aligned_spans_item = []
            try:
                for l, r, t, state, node in all_span:
                    if t == "p":
                        aligned_spans_item.append(pt_span[node])
                    else:
                        if node is None:
                            aligned_spans_item.append(None)
                        else:
                            aligned_spans_item.append(nt_span[node])
            except IndexError:
                print("bad alignment")
            aligned_spans.append(aligned_spans_item)
        return out, aligned_spans, pred.pt_nodes, pred.nt_nodes

    def generate(self, pred: TgtParserPrediction, **kwargs):
        assert pred.sampler is not None
        preds = pred.sampler()

        if self.use_copy:
            preds = self.expand_preds_using_copy(
                pred.src,
                preds,
                pred.pt_nodes,
                pred.nt_nodes,
                self.generation_max_length,
            )
        else:
            preds = self.expand_preds_not_using_copy(pred.src, preds)

        if self.generation_criteria == "ppl":
            preds = self.choose_samples_by_ppl(preds, pred, **kwargs)
        elif self.generation_criteria == "likelihood":
            preds = self.choose_samples_by_likelihood(preds, pred, **kwargs)
        elif self.generation_criteria == "contrastive":
            preds = self.choose_samples_by_constrastive(preds, pred, **kwargs)
        elif self.generation_criteria == "none":
            preds = [({"tgt": [item["tgt"] for item in batch]}, None) for batch in preds]
        return preds

    def expand_preds_using_copy(self, src, preds, pt_spans, nt_spans, max_len):
        # expand copied spans and build copy_position
        preds_ = []
        vocab = self.datamodule.tgt_vocab
        for batch, pt_spans_item, nt_spans_item, src_item in zip(preds, pt_spans, nt_spans, src):
            expanded_batch = []
            added = set()
            for item in batch:
                expanded = []
                for v, t in zip(item[0], item[1]):
                    if t == TokenType.VOCAB:
                        if len(expanded) + 1 > max_len:
                            break
                        expanded.append(vocab.convert_ids_to_tokens(v))
                    elif t == TokenType.COPY_PT:
                        span = pt_spans_item[v]
                        tokens = src_item[span[0] : span[1]]
                        if len(expanded) + len(tokens) > max_len:
                            break
                        expanded.extend(tokens)
                    elif t == TokenType.COPY_NT:
                        span = nt_spans_item[v]
                        tokens = src_item[span[0] : span[1]]
                        if len(expanded) + len(tokens) > max_len:
                            break
                        expanded.extend(tokens)

                assert len(expanded) <= max_len, (len(expanded), max_len)
                if (key := " ".join(expanded)) in added:
                    continue
                else:
                    added.add(key)
                pair = {"id": len(expanded_batch), "src": src_item, "tgt": expanded}
                expanded_batch.append(pair)
            expanded_batch = self.datamodule.process_pair(expanded_batch)
            expanded_batch = self.datamodule.apply_vocab(expanded_batch)
            preds_.append(expanded_batch)
        return preds_

    def expand_preds_not_using_copy(self, src, preds):
        preds_ = []
        for batch, src_item in zip(preds, src):
            batch = [item[0] for item in batch]
            batch = self.datamodule.tgt_vocab.convert_ids_to_tokens(batch)
            expanded_batch = []
            added = set()
            for item in batch:
                if (key := " ".join(item)) in added:
                    continue
                else:
                    added.add(key)
                expanded_batch.append({"id": len(expanded_batch), "src": src_item, "tgt": item})
            expanded_batch = self.datamodule.process_pair(expanded_batch)
            expanded_batch = self.datamodule.apply_vocab(expanded_batch)
            preds_.append(expanded_batch)
        return preds_

    @torch.no_grad()
    def choose_samples_by_ppl(self, preds, pred: TgtParserPrediction, **kwargs):

        # JUST FOR ANALYSIS. REMOVE IF REPORTING FINAL RESULTS
        tgt_lens = kwargs.get("tgt_lens")

        new_preds = []
        pred = copy(pred).clear()
        for bidx, preds_batch in enumerate(preds):
            batch_size = pred.batch_size if self.generation_ppl_batch_size is None else self.generation_ppl_batch_size
            loader = DataLoader(dataset=preds_batch, batch_size=batch_size, collate_fn=self.datamodule.collator)
            ppl = np.full((len(preds_batch),), 1e9)

            for batch in loader:
                batch = self.datamodule.transfer_batch_to_device(batch, pred.device, 0)
                observed = {
                    "x": batch["tgt_ids"],
                    "lengths": batch["tgt_lens"],
                    "pt_copy": batch.get("copy_token"),
                    "nt_copy": batch.get("copy_phrase"),
                }
                sub_pred = pred.get_and_expand(bidx, len(batch["tgt_ids"]))
                sub_pred = self.observe_x(sub_pred, **observed)
                nll = sub_pred.dist.nll.detach().cpu().numpy()
                ppl_batch = np.exp(nll / np.array(batch["tgt_lens"]))
                for i, ppl_item in zip(batch["id"].tolist(), ppl_batch):
                    ppl[i] = ppl_item

                if tgt_lens is not None:
                    for i, cl, l in zip(batch["id"].tolist(), batch["tgt_lens"], tgt_lens):
                        if cl != l:
                            ppl[i] += 1000

            assert not np.any(np.isnan(ppl))
            chosen = np.argmin(ppl)
            if ppl[chosen] > 1e6:
                logger.warning(f"The minimum ppl is {ppl[chosen]}")
            new_preds.append((preds_batch[chosen], ppl[chosen]))
        return new_preds

    @torch.no_grad()
    def choose_samples_by_likelihood(self, preds, pred: TgtParserPrediction, **kwargs):
        new_preds = []
        pred = copy(pred).clear()
        for bidx, preds_batch in enumerate(preds):
            batch_size = pred.batch_size if self.generation_ppl_batch_size is None else self.generation_ppl_batch_size
            loader = DataLoader(dataset=preds_batch, batch_size=batch_size, collate_fn=self.datamodule.collator)
            nll = np.full((len(preds_batch),), 1e9)

            for batch in loader:
                batch = self.datamodule.transfer_batch_to_device(batch, pred.device, 0)

                observed = {
                    "x": batch["tgt_ids"],
                    "lengths": batch["tgt_lens"],
                    "pt_copy": batch.get("copy_token"),
                    "nt_copy": batch.get("copy_phrase"),
                }
                sub_pred = pred.get_and_expand(bidx, len(batch["tgt_ids"]))
                sub_pred = self.observe_x(sub_pred, **observed)
                nll_batch = sub_pred.dist.nll.detach().cpu().numpy()

                for i, nll_item in zip(batch["id"].tolist(), nll_batch):
                    nll[i] = nll_item

            assert not np.any(np.isnan(nll))
            chosen = np.argmin(nll)
            if nll[chosen] > 1e6:
                logger.warning(f"The minimum nll is {nll[chosen]}")
            new_preds.append((preds_batch[chosen], nll[chosen]))
        return new_preds

    @torch.no_grad()
    def choose_samples_by_constrastive(self, preds, pred: TgtParserPrediction, baseline_model, **kwargs):
        new_preds = []
        pred = copy(pred).clear()
        for bidx, preds_batch in enumerate(preds):
            batch_size = pred.batch_size if self.generation_ppl_batch_size is None else self.generation_ppl_batch_size
            loader = DataLoader(dataset=preds_batch, batch_size=batch_size, collate_fn=self.datamodule.collator)
            criteria = np.full((len(preds_batch),), 1e9)

            for batch in loader:
                batch = self.datamodule.transfer_batch_to_device(batch, pred.device, 0)

                observed = {
                    "x": batch["tgt_ids"],
                    "lengths": batch["tgt_lens"],
                    "pt_copy": batch.get("copy_token"),
                    "nt_copy": batch.get("copy_phrase"),
                }
                sub_pred = pred.get_and_expand(bidx, len(batch["tgt_ids"]))
                sub_pred = self.observe_x(sub_pred, **observed)
                nll = sub_pred.dist.nll.detach().cpu().numpy()
                ppl_batch = np.exp(nll / np.array(batch["tgt_lens"]))

                nll2 = baseline_model(batch["tgt_ids"], batch["tgt_lens"]).nll.detach().cpu().numpy()
                ppl_batch_baseline = np.exp(nll2 / np.array(batch["tgt_lens"]))

                ppl_batch = ppl_batch - ppl_batch_baseline

                for i, ppl_item in zip(batch["id"].tolist(), ppl_batch):
                    criteria[i] = ppl_item

            assert not np.any(np.isnan(criteria))
            chosen = np.argmin(criteria)
            if criteria[chosen] > 1e6:
                logger.warning(f"The minimum criteria is {criteria[chosen]}")
            new_preds.append((preds_batch[chosen], criteria[chosen]))
        return new_preds

    def to_str_tokens(self, preds, src):
        tgt_vocab = self.datamodule.tar
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
                s_len = s[1] - s[0]
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
        num_terms = sum(item[0] + 1 == item[1] for item in pt_spans)
        # there must be something in nt_spans
        assert len(nt_spans) > 0
        # root must be the last of nt_spans
        assert nt_spans[-1][0] == 0 and nt_spans[-1][1] == num_terms
        # singles must be ordered and placed at the begining of pt_spans
        for i, item in zip(range(num_terms), pt_spans):
            assert item[0] == item[1] - 1 == i

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
        pt_align=None,
        observed_mask=None,
        prior_alignment=None,
    ):
        batch_size, n = tgt.shape[:2]
        term = term.unsqueeze(1).expand(batch_size, n, pt, term.size(2))
        tgt_expand = tgt.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        term = torch.gather(term, 3, tgt_expand).squeeze(3)

        constraint_scores = None
        lse_scores = None
        add_scores = None

        if self.use_copy:
            # TODO mask COPY if pt_copy and nt_copy is not given.
            if pt_copy is not None:
                term = self.build_pt_copy_constraint(term, pt_copy)
            if pt_align is not None:
                # TODO refactor this
                term = self.build_pt_align_constraint(term, pt_align)

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

        if observed_mask is not None and self.use_observed:
            constraint_scores = self.build_observed_span_constraint(
                batch_size, n, max_nt_spans, observed_mask, constraint_scores
            )

        if prior_alignment is not None:
            raise NotImplementedError
            term = self.build_pt_prior_alignment_soft_constraint(term, prior_alignment)

        return {"term": term, "root": root, "constraint": constraint_scores, "lse": lse_scores, "add": add_scores}

    def build_pt_copy_constraint(self, terms, constraint):
        batch_size, n = terms.shape[:2]
        terms = terms.view(batch_size, n, self.pt_states, -1)
        copy_m = constraint[:, : terms.shape[-1]].transpose(1, 2)
        terms[:, :, -1].fill_(self.neg_huge)
        terms[:, :, -1, : copy_m.shape[2]] = self.neg_huge * ~copy_m
        terms = terms.view(batch_size, n, -1)
        return terms

    def build_pt_align_constraint(self, terms, constraint):
        batch_size, n = terms.shape[:2]
        terms = terms.view(batch_size, n, self.pt_states, -1)
        align_m = constraint[:, : terms.shape[-1]].transpose(1, 2)
        value = align_m.any(2, keepdim=True).float() * (1 - align_m.float()) * self.neg_huge
        terms[:, :, :-1, : align_m.shape[2]] += value.unsqueeze(2)
        terms = terms.view(batch_size, n, -1)
        return terms

    def build_pt_prior_alignment_soft_constraint(self, terms, constraint):
        batch_size, n = terms.shape[:2]
        terms = terms.clone()
        terms = terms.view(batch_size, n, self.pt_states, -1)
        constraint = constraint[:, : terms.shape[-1]].transpose(1, 2)
        terms[..., : constraint.shape[2]] *= constraint.unsqueeze(2)
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
            for i, (l, r, tag) in enumerate(nt_spans_inst):
                if tag == NO_COPY_SPAN:
                    continue
                w = r - l - 2
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
            constraint = self.post_process_nt_constraint(
                self.get_init_nt_constraint(batch_size, n, max_nt_spans), observed_constraint[0].device
            )
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
            "strict": self.generation_strict,
            "unk": 1,
        }
