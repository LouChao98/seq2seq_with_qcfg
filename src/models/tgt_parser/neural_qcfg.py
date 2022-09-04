import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_struct.distributions import SentCFG

from ...datamodules.components.vocab import VocabularyPair
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .struct.pcfg import PCFG, TokenType

log = logging.getLogger(__file__)


class NeuralQCFGTgtParser(TgtParserBase):
    def __init__(
        self,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        nt_states=10,
        pt_states=1,
        rule_constraint_type=0,
        rule_constraint_method="hard",
        rule_constraint_method_args=None,
        use_copy=False,
        nt_span_range=[2, 1000],
        pt_span_range=[1, 1],
        num_samples=10,
        check_ppl=False,
        check_ppl_batch_size=None,
    ):
        super(NeuralQCFGTgtParser, self).__init__()
        self.neg_huge = -1e9
        self.pcfg = PCFG()
        self.vocab = vocab
        self.dim = dim
        self.src_dim = src_dim
        self.num_layers = num_layers
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.rule_constraint_type = rule_constraint_type
        self.rule_constraint_method = rule_constraint_method
        self.rule_constraint_method_args = rule_constraint_method_args
        self.use_copy = use_copy
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range
        self.num_samples = num_samples
        self.check_ppl = check_ppl  # If true, there is a length limitation.
        self.check_ppl_batch_size = check_ppl_batch_size

        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.src_nt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.src_pt_node_mlp = MultiResidualLayer(src_dim, dim, num_layers=num_layers)
        self.rule_mlp_parent = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_left = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.rule_mlp_right = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(
            dim, dim, out_dim=vocab, num_layers=num_layers
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.src_nt_emb.data)
        nn.init.xavier_uniform_(self.src_pt_emb.data)

    def forward(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params, *_ = self.get_params(node_features, spans, x, **kwargs)
        out = self.pcfg(params, lengths, False)  # nll
        return out

    def forward_pr(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params, *_ = self.get_params(node_features, spans, x, **kwargs)
        return self.get_pr_term(params, lengths)

    def parse(self, x, lengths, node_features, spans, **kwargs):
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans, x, **kwargs
        )
        out = self.pcfg(params, lengths, True)

        # out: list of list, containing spans (i, j, label)
        src_nt_states = self.nt_states * nt_num_nodes
        src_pt_states = self.pt_states * pt_num_nodes
        all_spans_node = []
        for b, (all_span, pt_span, nt_span) in enumerate(zip(out, pt_spans, nt_spans)):
            all_span_node = []
            for l, r, label in all_span:
                if l == r:
                    if label < src_pt_states:
                        all_span_node.append(pt_span[label % pt_num_nodes])
                    else:
                        # these are for tgt_nt_states, which are removed for now.
                        all_span_node.append([-1, -1, label - src_pt_states])
                else:
                    if label < src_nt_states:
                        all_span_node.append(nt_span[label % nt_num_nodes])
                    else:
                        all_span_node.append([-1, -1, label - src_nt_states])
            all_spans_node.append(all_span_node)
        return out, all_spans_node, pt_spans, nt_spans

    def generate(
        self,
        node_features,
        spans,
        vocab_pair: VocabularyPair,
        src_ids: torch.Tensor,
        src: List[List[str]],
    ):
        # if check_ppl=True, I will compute ppl for samples, return the one with minimum ppl
        # else, just return the one with the maximum score

        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans
        )

        max_len = 40
        preds = self.pcfg.sampled_decoding(
            params,
            nt_spans,
            self.nt_states,
            pt_spans,
            self.pt_states,
            num_samples=self.num_samples,
            use_copy=self.use_copy,
            max_length=max_len,
        )

        # expand copied spans and build copy_position
        preds_ = []
        copy_positions = []
        device = node_features[0].device
        src_lens = (src_ids != vocab_pair.src.pad_token_id).sum(1).tolist()
        src_ids = src_ids.tolist()
        for batch, pt_spans_inst, nt_spans_inst, src_ids_inst, src_len in zip(
            preds, pt_spans, nt_spans, src_ids, src_lens
        ):
            if self.use_copy:
                expanded_batch = []
                copy_pts = []
                copy_nts = []
                copy_unks = []
                for inst in batch:
                    expanded = []
                    copy_pt = np.zeros((src_len, max_len), dtype=np.bool8)
                    copy_nt = [[] for _ in range(max_len)]  # no need to prune
                    copy_unk = {}  # record position if copy unk token
                    for v, t in zip(inst[0], inst[1]):
                        if t == TokenType.VOCAB:
                            if len(expanded) + 1 > max_len:
                                break
                            expanded.append(v)
                        elif t == TokenType.COPY_PT:
                            span = pt_spans_inst[v]
                            tokens = vocab_pair.src2tgt(
                                src_ids_inst[span[0] : span[1] + 1]
                            )
                            if len(expanded) + len(tokens) > max_len:
                                break
                            copy_pt[span[0], len(expanded)] = True
                            if tokens[0] == vocab_pair.tgt.unk_token_id:
                                copy_unk[len(expanded)] = span[0]
                            expanded.extend(tokens)
                        elif t == TokenType.COPY_NT:
                            span = nt_spans_inst[v]
                            tokens = vocab_pair.src2tgt(
                                src_ids_inst[span[0] : span[1] + 1]
                            )
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
                        continue
                        assert False, "Debug this"
                    if len(expanded) > max_len:
                        continue
                    expanded_batch.append((expanded, inst[2]))
                    copy_pts.append(copy_pt)
                    copy_nts.append(copy_nt)
                    copy_unks.append(copy_unk)
                copy_pts = torch.from_numpy(np.stack(copy_pts, axis=0)).to(device)
                copy_positions.append((copy_pts, copy_nts, copy_unks))
                preds_.append(expanded_batch)
            else:
                expanded_batch = []
                for inst in batch:
                    expanded_batch.append((inst[0], inst[2]))
                copy_positions.append((None, None, None))
                preds_.append(expanded_batch)

        preds = preds_

        if self.check_ppl:
            padid = vocab_pair.tgt.pad_token_id or 0
            new_preds = []
            for i, (preds_one_inp, (copy_pt, copy_nt, copy_unk)) in enumerate(
                zip(preds, copy_positions)
            ):
                to_keep = [1 < len(inst[0]) <= 60 for inst in preds_one_inp]
                _ids = [inst[0] for inst, flag in zip(preds_one_inp, to_keep) if flag]

                sort_id = list(range(len(_ids)))
                sort_id.sort(key=lambda x: len(_ids[x]), reverse=True)
                _ids = [_ids[i] for i in sort_id]
                _lens = [len(inst) for inst in _ids]
                _ids_t = torch.full((len(_ids), _lens[0]), padid)
                for j, (snt, length) in enumerate(zip(_ids, _lens)):
                    _ids_t[j, :length] = torch.tensor(snt)
                _ids_t = _ids_t.to(node_features[0].device)

                if copy_pt is not None:
                    copy_pt = copy_pt[to_keep]
                    copy_pt = copy_pt[sort_id]
                    copy_nt = [item for item, flag in zip(copy_nt, to_keep) if flag]
                    copy_nt = [copy_nt[i] for i in sort_id]
                    copy_unk = [item for item, flag in zip(copy_unk, to_keep) if flag]
                    copy_unk = [copy_unk[i] for i in sort_id]

                batch_size = (
                    len(node_features)
                    if self.check_ppl_batch_size is None
                    else self.check_ppl_batch_size
                )
                ppl = []
                for j in range(0, len(_ids), batch_size):
                    real_batch_size = min(batch_size, len(_ids) - j)
                    _node_ft = [node_features[i] for _ in range(real_batch_size)]
                    _spans = [spans[i] for _ in range(real_batch_size)]
                    if copy_pt is not None:
                        _copy = (
                            copy_pt[j : j + batch_size, :, : _ids_t.shape[1]],
                            copy_nt[j : j + batch_size],
                        )
                    else:
                        _copy = None
                    max_len = max(_lens[j : j + batch_size])
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
                chosen = np.argmin(ppl)
                new_preds.append(
                    (
                        _ids[chosen],
                        ppl[chosen],
                        None if copy_pt is None else copy_unk[chosen],
                    )
                )
            preds = new_preds
        else:
            assert False, "Bad impl of score. see sample()."
            preds_ = []
            for item in preds:
                item = max(item, key=lambda x: x[1])
                preds_.append(item)
            preds = preds_

        pred_strings = []
        for pred, src_sent in zip(preds, src):
            snt, score, copy_unk = pred
            try:
                sent = vocab_pair.tgt.convert_ids_to_tokens(snt)
                if copy_unk is not None:
                    for t, s in copy_unk.items():
                        sent[t] = src_sent[s]
                pred_strings.append((sent, score))
            except IndexError:
                print("Bad pred:", snt)
                pred_strings.append([("", -999)])
        return pred_strings

    def get_params(
        self,
        node_features,
        spans,
        x: Optional[torch.Tensor] = None,
        copy_position=None,  # (pt, nt)
        impossible_span_mask=None,
        ignore_src=False,
    ):
        if copy_position is None or not self.use_copy:
            copy_position = (None, None)

        batch_size = len(spans)

        (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
        ) = self.build_src_features(spans, node_features)
        device = nt_node_features.device

        # e = u + h
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        if ignore_src:
            src_nt_node_emb.zero_()
        src_nt_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb = src_nt_emb

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        if ignore_src:
            src_pt_node_emb.zero_()
        src_pt_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        src_pt_emb = src_pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
        pt_emb = src_pt_emb

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        mask = (
            torch.arange(nt_num_nodes, device=device)
            .view(1, 1, -1)
            .expand(batch_size, 1, -1)
        )
        allowed = (torch.tensor(nt_num_nodes_list, device=device) - 1).view(-1, 1, 1)
        roots = torch.where(mask == allowed, roots, roots.new_full((1,), self.neg_huge))
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_emb)
        rule_emb_right = self.rule_mlp_right(all_emb)

        rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        rule_emb_child = rule_emb_child.view(batch_size, (nt + pt) ** 2, self.dim)
        rules = (
            torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
            .log_softmax(-1)
            .clone()
        )
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=rules.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=rules.device).unsqueeze(1)
        lhs_mask = nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        _pt_rhs_mask = pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        # fmt: on
        rhs_mask = torch.cat([lhs_mask, _pt_rhs_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", lhs_mask, rhs_mask, rhs_mask)
        rules[~mask] = self.neg_huge

        constraint_mask = None
        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                constraint_mask = self.get_rules_mask1(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 2:
                constraint_mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            if self.rule_constraint_method == "hard":
                rules[~constraint_mask] = self.neg_huge

        # rules = (
        #     rules.view(batch_size, nt, (nt + pt) ** 2)
        #     .log_softmax(2)
        #     .view(batch_size, nt, nt + pt, nt + pt)
        # )

        # A->a
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)
        copy_nt = None

        if x is not None:
            n = x.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if copy_position[0] is not None:
                # TODO sanity check: pt_spans begin with (0,0), (1,1) ... (n-1,n-1)
                terms = terms.view(batch_size, n, self.pt_states, -1)
                terms[:, :, -1] = (
                    0.1 * self.neg_huge * ~copy_position[0].transpose(1, 2)
                )
                terms = terms.view(batch_size, n, -1)
            if copy_position[1] is not None:
                # mask=True will set to value
                copy_nt = [
                    np.full(
                        (batch_size, n - w, self.nt_states, nt_num_nodes),
                        self.neg_huge,
                        dtype=np.float32,
                    )
                    for w in range(1, n)
                ]
                for batch_idx, (nt_spans_inst, possible_copy) in enumerate(
                    zip(nt_spans, copy_position[1])
                ):
                    for i, (l, r, _) in enumerate(nt_spans_inst):
                        w = r - l - 1
                        t = None
                        if w >= len(possible_copy):
                            continue
                        for possible_s, possible_t in possible_copy[w]:
                            if possible_s == l:
                                t = possible_t
                                break
                        if t is not None:
                            copy_nt[w][batch_idx, t, -1, i] = 0
                copy_nt_ = []
                # TODO mask can use expand
                for item in copy_nt:
                    mask = np.zeros_like(item, dtype=np.bool8)
                    mask[:, :, -1] = True
                    item = torch.from_numpy(item.reshape(item.shape[:2] + (-1,)))
                    mask = torch.from_numpy(mask.reshape(item.shape))
                    copy_nt_.append((item.to(terms.device), mask.to(terms.device)))
                copy_nt = copy_nt_
            if impossible_span_mask is not None:
                assert copy_position[1] is None, "Not implemented"
                copy_nt_ = []  # TODO rename to some meaningful name
                neg_huge = rules.new_tensor(self.neg_huge)
                for item in impossible_span_mask:
                    copy_nt_.append(
                        (
                            neg_huge,
                            item.unsqueeze(-1),
                        )
                    )
                copy_nt = copy_nt_

        # TODO return everything in dict, or some thing else
        params = {
            "term": terms,
            "root": roots,
            "rule": rules,
            "copy_nt": copy_nt,
            "constraint_mask": constraint_mask,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def build_src_features(self, spans, node_features):

        # seperate nt and pt features according to span width
        # TODO sanity check: the root node must be the last element of nt_spans.
        #      because of root rules' masking. single words are also required to
        #      be placed in ascending order.
        # NOTE TreeLSTM guarantees this.
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for spans_inst, node_features_inst in zip(spans, node_features):

            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []

            # indices = list(range(len(spans_inst)))
            # shuffle_start = (len(spans_inst) + 1) // 2
            # shuffle_parts = indices[shuffle_start:-1]
            # random.shuffle(shuffle_parts)
            # indices = indices[:shuffle_start] + shuffle_parts + indices[-1:]
            # spans_inst = [spans_inst[i] for i in indices]
            # node_features_inst = [node_features_inst[i] for i in indices]

            for s, f in zip(spans_inst, node_features_inst):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(f)
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(f)
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_features_inst[-1])
                nt_span.append(spans_inst[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_num_nodes_list = [len(inst) for inst in nt_node_features]
        pt_num_nodes_list = [len(inst) for inst in pt_node_features]
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

    def get_pr_term(self, params, lens):
        if self.rule_constraint_type in (1, 2, 3):
            return self.get_pr_term_by_line_search(params, lens)
        raise NotImplementedError

    def get_pr_term_by_line_search(self, params, lens):
        # E[constraint] <= b, constraint_mask 1=allow
        constraint = 1.0 - params["constraint_mask"].float()
        lambdas = self.solve_line_search(
            params,
            lens,
            constraint,
            torch.full(
                (len(params["rule"]),),
                self.rule_constraint_method_args.b,
                device=params["rule"].device,
            ),
            self.rule_constraint_method_args.lbound,
            self.rule_constraint_method_args.rbound,
            self.rule_constraint_method_args.num_point,
            self.rule_constraint_method_args.num_iter,
        )
        constrained_params = (
            params["term"].detach(),
            params["rule"].detach() - constraint * lambdas.view(-1, 1, 1, 1),
            params["root"].detach(),
            params["copy_nt"],
        )
        original_params = (
            params["term"],
            params["rule"],
            params["root"],
            params["copy_nt"],
        )
        q = SentCFG(constrained_params, lens)
        q_margin = q.marginals
        p = SentCFG(original_params, lens)
        ce = (
            p.partition
            - (q_margin[0].detach() * original_params[0]).sum((1, 2))
            - (q_margin[1].detach() * original_params[1]).sum((1, 2, 3))
            - (q_margin[2].detach() * original_params[2]).sum(1)
        )
        return ce

    @torch.no_grad()
    def solve_line_search(
        self,
        params,
        lens,
        factorized_constraint,
        b,
        lbound=1e-4,
        rbound=1e4,
        num_point=16,
        num_iter=3,
    ):
        batch_size = len(params["rule"])
        lambdas = []
        for bidx in range(batch_size):
            lb, rb = lbound, rbound
            params_inst = {
                "term": params["term"][bidx, None].detach().expand(num_point, -1, -1),
                "root": params["root"][bidx, None].detach().expand(num_point, -1),
                "copy_nt": (
                    [
                        (
                            v[bidx, None].expand(num_point, -1, -1),
                            m[bidx, None].expand(num_point, -1, -1),
                        )
                        for v, m in params["copy_nt"]
                    ]
                    if params["copy_nt"] is not None
                    else None
                ),
            }
            constraint_inst = factorized_constraint[bidx, None]
            b_inst = b[bidx, None]
            lt, rt, max_t, max_l = None, None, None, None
            for itidx in range(num_iter):
                if itidx > 0:  # skip lb rb
                    lgrid_np = np.geomspace(lb, rb, num_point + 2, dtype=np.float32)
                    lgrid = torch.from_numpy(lgrid_np[1:-1]).to(constraint_inst.device)
                else:
                    lgrid_np = np.geomspace(lb, rb, num_point, dtype=np.float32)
                    lgrid = torch.from_numpy(lgrid_np).to(constraint_inst.device)
                # potential * exp(-lambda * constraint)
                rule_inst = params["rule"][
                    bidx, None
                ].detach() - constraint_inst * lgrid.view(
                    -1, *[1] * (constraint_inst.ndim - 1)
                )
                params_inst["rule"] = rule_inst
                target = -lgrid * b_inst + self.pcfg(
                    params_inst, [lens[bidx]] * num_point
                )
                target = target.cpu().numpy()
                if itidx > 0:  # take back lb, rb and argmax
                    target = [lt, *target.tolist(), rt]
                    max_insert = lgrid_np.searchsorted(max_l)
                    lgrid_np = lgrid_np.tolist()
                    lgrid_np.insert(max_insert, max_l)
                    target.insert(max_insert, max_t)
                    target = np.asarray(target)
                argmax_i = np.argmax(target)
                if argmax_i == 0 or argmax_i == len(target) - 1:
                    lambdas.append(lgrid_np[argmax_i])
                    if itidx == 0 and argmax_i != 0:
                        # A very small i (argmax=0) is acceptable as it means
                        # we can satisfy the constraint without effort
                        #
                        # This can also be understood as the slack-penalty version
                        # of PR. The dual problem is
                        # max -b\lambda - log Z s.t. 0 <= \lambda, dual_norm(\lambda) < \sigma
                        log.warning("Line search fails.")
                    break
                lt, rt = target[argmax_i - 1], target[argmax_i + 1]
                lb, rb = lgrid_np[argmax_i - 1], lgrid_np[argmax_i + 1]
                max_t = target[argmax_i]
                max_l = lgrid_np[argmax_i]
                if rb - lb < 1e-3:
                    lambdas.append(lgrid_np[argmax_i])
                    break
            else:
                # any possible for argmax_i not defined before reference?
                lambdas.append(lgrid_np[argmax_i])
        return torch.tensor(lambdas, device=params["rule"].device, dtype=torch.float32)

    def get_rules_mask1(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return True for not masked
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        nt_node_mask = torch.ones(
            batch_size, nt_num_nodes, nt_num_nodes, dtype=torch.bool
        )
        pt_node_mask = torch.ones(
            batch_size, nt_num_nodes, pt_num_nodes, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        # TODO vectorization
        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j] = False
                for j, child_span in enumerate(pt_span):
                    if not (is_parent(parent_span, child_span)):
                        pt_node_mask[b, i, j] = False

        nt_node_mask = (
            nt_node_mask[:, None, :, None, :]
            .expand(
                batch_size,
                self.nt_states,
                nt_num_nodes,
                self.nt_states,
                nt_num_nodes,
            )
            .contiguous()
        )
        pt_node_mask = (
            pt_node_mask[:, None, :, None, :]
            .expand(
                batch_size,
                self.nt_states,
                nt_num_nodes,
                self.pt_states,
                pt_num_nodes,
            )
            .contiguous()
        )

        nt_node_mask = nt_node_mask.view(batch_size, nt, nt)
        pt_node_mask = pt_node_mask.view(batch_size, nt, pt)
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        bsz = batch_size
        src_nt = self.nt_states
        src_pt = self.pt_states
        node_nt = nt_num_nodes
        node_pt = pt_num_nodes
        node_mask = torch.zeros(
            bsz,
            src_nt * node_nt,
            src_nt * node_nt + src_pt * node_pt,
            src_nt * node_nt + src_pt * node_pt,
            dtype=torch.bool,
            device=device,
        )

        nt_idx = slice(0, src_nt * node_nt)
        pt_idx = slice(src_nt * node_nt, src_nt * node_nt + src_pt * node_pt)

        nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_nt, node_nt
        )
        nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_pt, node_pt
        )
        nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_nt, node_nt
        )
        nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_pt, node_pt
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        def is_strict_parent(parent, child):
            return is_parent(parent, child) and parent != child

        def span_len(span):
            return span[1] - span[0] + 1

        def covers(parent, child1, child2):
            return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
                (parent[0] == child1[0] and parent[1] == child2[1])
                or (parent[0] == child2[0] and parent[1] == child1[1])
            )

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    nt_ntnt[b, :, i, :, i, :, i].fill_(True)
                    for j, child in enumerate(pt_span):
                        if is_strict_parent(parent, child):
                            nt_ntpt[b, :, i, :, i, :, j].fill_(True)
                            nt_ptnt[b, :, i, :, j, :, i].fill_(True)
                if span_len(parent) == 1:
                    for j, child in enumerate(pt_span):
                        if parent == child:
                            nt_ptnt[b, :, i, :, j, :, i].fill_(True)
                            nt_ntpt[b, :, i, :, i, :, j].fill_(True)
                            nt_ptpt[b, :, i, :, j, :, j].fill_(True)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            nt_ntnt[b, :, i, :, j, :, k].fill_(True)
                            nt_ntnt[b, :, i, :, k, :, j].fill_(True)
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ntpt[b, :, i, :, j, :, k].fill_(True)
                            nt_ptnt[b, :, i, :, k, :, j].fill_(True)
                for j, child1 in enumerate(pt_span):
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ptpt[b, :, i, :, j, :, k].fill_(True)
                            nt_ptpt[b, :, i, :, k, :, j].fill_(True)

        return node_mask.view(batch_size, nt, nt + pt, nt + pt)
