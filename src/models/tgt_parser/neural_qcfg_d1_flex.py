from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...datamodules.components.vocab import VocabularyPair
from ...utils.fn import spans2tree
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.d1_pcfg_flex import D1PCFGFlex
from .struct.pcfg import PCFG, TokenType


def get_nn(dim, cpd_rank):
    return nn.Sequential(
        nn.LeakyReLU(), nn.Linear(dim, cpd_rank)  # , nn.LayerNorm(cpd_rank)
    )


@torch.jit.script
def normalize(t: torch.Tensor):
    t = t - t.amax((-2, -1), keepdim=True)
    t = t.exp()
    t = t / (t.sum((-2, -1), keepdim=True) + 1e-9)
    return t


class NeuralQCFGD1FlexTgtParser(NeuralQCFGTgtParser):
    def __init__(self, cpd_rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.nt_states == self.pt_states
        dim = self.dim
        num_layers = self.num_layers
        self.cpd_rank = cpd_rank
        self.pcfg = D1PCFGFlex(self.nt_states, self.pt_states)
        # self.root_mlp_i = MultiResidualLayer(dim, dim, num_layers=num_layers)
        # self.root_mlp_j = MultiResidualLayer(dim, dim, num_layers=num_layers)
        # self.root_mlp_k = MultiResidualLayer(dim, dim, num_layers=num_layers)
        self.root_mlp_i = get_nn(dim, dim)
        self.root_mlp_j = get_nn(dim, dim)
        self.root_mlp_k = get_nn(dim, dim)
        self.rijk_weight = nn.Parameter(torch.empty(cpd_rank, dim, dim))
        nn.init.kaiming_uniform_(self.rijk_weight)
        self.ai_r_nn = get_nn(dim, cpd_rank)
        self.r_b_nn = get_nn(dim, cpd_rank)
        self.r_c_nn = get_nn(dim, cpd_rank)
        self.r_jk_nn = get_nn(dim, cpd_rank)

    def forward(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params, *_ = self.get_params(node_features, spans, x, **kwargs)
        out = self.pcfg(params, lengths, False)
        return out

    def parse(self, x, lengths, node_features, spans, params=None, **kwargs):
        if params is None:
            params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
                node_features, spans, x, **kwargs
            )
        else:
            params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = params

        # params2 = D1PCFG.get_pcfg_rules(params, self.nt_states)
        # out = PCFG()(params2, lengths, decode=True)
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

        # preds = PCFG().sampled_decoding(
        #     self.pcfg.get_pcfg_rules({**params}, self.nt_states),
        #     nt_spans,
        #     self.nt_states,
        #     pt_spans,
        #     self.pt_states,
        #     num_samples=self.num_samples,
        #     use_copy=self.use_copy,
        #     max_length=max_len,
        # )

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

                    if len(expanded) > max_len:
                        assert False
                    expanded_batch.append(expanded)
                    copy_pts.append(copy_pt)
                    copy_nts.append(copy_nt)
                    copy_unks.append(copy_unk)
                copy_pts = torch.from_numpy(np.stack(copy_pts, axis=0)).to(device)
                copy_positions.append((copy_pts, copy_nts, copy_unks))
                preds_.append(expanded_batch)
            else:
                expanded_batch = []
                for inst in batch:
                    expanded_batch.append(inst[0])
                copy_positions.append((None, None, None))
                preds_.append(expanded_batch)

        preds = preds_

        if self.check_ppl:
            with torch.no_grad():
                padid = vocab_pair.tgt.pad_token_id or 0
                new_preds = []
                for i, (preds_one_inp, (copy_pt, copy_nt, copy_unk)) in enumerate(
                    zip(preds, copy_positions)
                ):
                    to_keep = [1 < len(inst) <= 60 for inst in preds_one_inp]
                    _ids = [inst for inst, flag in zip(preds_one_inp, to_keep) if flag]
                    # TODO
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
                        copy_unk = [
                            item for item, flag in zip(copy_unk, to_keep) if flag
                        ]
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
        nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        nt_state_emb = self.src_nt_emb.expand(batch_size, self.nt_states, self.dim)
        nt_emb = nt_state_emb.unsqueeze(2) + nt_node_emb.unsqueeze(1)

        pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        pt_state_emb = self.src_pt_emb.expand(batch_size, self.pt_states, self.dim)
        pt_emb = pt_state_emb.unsqueeze(2) + pt_node_emb.unsqueeze(1)

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
        state_emb = torch.cat([nt_state_emb, pt_state_emb], 1)
        node_emb = torch.cat([nt_node_emb, pt_node_emb], 1)
        rule_head = self.ai_r_nn(
            self.rule_mlp_parent(nt_emb.view(batch_size, -1, self.dim))
        ).softmax(-1)
        rule_left = (
            self.r_b_nn(self.rule_mlp_left(state_emb)).transpose(1, 2).softmax(-1)
        )
        rule_right = (
            self.r_c_nn(self.rule_mlp_right(state_emb)).transpose(1, 2).softmax(-1)
        )

        i = self.root_mlp_i(nt_node_emb)
        j = self.root_mlp_j(node_emb)
        k = self.root_mlp_k(node_emb)
        rule_slr = torch.einsum(
            "rab,xia,xjkb->xrijk",
            self.rijk_weight,
            F.leaky_relu(i),
            F.leaky_relu(j[:, :, None] + k[:, None, :]),
        )  # softmax

        new_slr = torch.empty_like(rule_slr)
        nnn = nt_num_nodes
        new_slr[..., :nnn, :nnn] = normalize(rule_slr[..., :nnn, :nnn])
        new_slr[..., :nnn, nnn:] = normalize(rule_slr[..., :nnn, nnn:])
        new_slr[..., nnn:, :nnn] = normalize(rule_slr[..., nnn:, :nnn])
        new_slr[..., nnn:, nnn:] = normalize(rule_slr[..., nnn:, nnn:])
        rule_slr = new_slr

        # ijk = i[:, :, None, None] + j[:, None, :, None] + k[:, None, None, :]
        # num_nodes = nt_num_nodes  # + pt_num_nodes
        # rule_slr = (
        #     self.r_jk_nn(ijk)
        #     .movedim(4, 1)
        #     .view(batch_size, self.cpd_rank, num_nodes, -1)
        #     .log_softmax(-1)
        #     .view(batch_size, self.cpd_rank, num_nodes, num_nodes, num_nodes)
        #     .clone()
        # )

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=i.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=i.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=i.device).unsqueeze(1)
        # fmt: on
        mask = torch.cat([nt_mask, pt_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", nt_mask, mask, mask)
        mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
        rule_slr[~mask] = 0  # = self.neg_huge

        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                mask = self.get_rules_mask1(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 2:
                mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 3:
                mask = self.get_rules_mask3(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            else:
                raise ValueError("Bad constraint_type")
            mask = mask.unsqueeze(1).expand(-1, self.cpd_rank, -1, -1, -1)
            rule_slr[~mask] = 0  # = self.neg_huge

        # A->a
        terms = self.vocab_out(pt_emb).log_softmax(-1)
        terms = terms.view(batch_size, -1, terms.shape[-1])

        copy_nt = None
        if x is not None:
            n = x.size(1)
            pt = terms.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if copy_position[0] is not None:
                terms = terms.view(batch_size, n, self.pt_states, -1).clone()
                # copy_position may be longer than terms
                #  and shorter than terms because terms are not shrinked
                #   when constructed on a subbatch
                copy_m = copy_position[0][:, : terms.shape[-1]].transpose(1, 2)
                terms[:, :, -1, : copy_m.shape[2]] = self.neg_huge * ~copy_m
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
                        if w >= len(possible_copy) or w < 0:
                            continue
                        for possible_s, possible_t in possible_copy[w]:
                            if possible_s == l:
                                t = possible_t
                                break
                        if t is not None:
                            copy_nt[w][batch_idx, t, -1, i] = 0
                copy_nt_ = []
                for item in copy_nt:
                    mask = np.zeros_like(item, dtype=np.bool8)
                    mask[:, :, -1] = True
                    item = torch.from_numpy(item)
                    mask = torch.from_numpy(mask)
                    copy_nt_.append((item.to(terms.device), mask.to(terms.device)))
                copy_nt = copy_nt_

            if impossible_span_mask is not None:
                assert copy_position[1] is None, "Not implemented"
                copy_nt_ = []  # TODO rename to some meaningful name
                neg_huge = terms.new_tensor(self.neg_huge)
                for item in impossible_span_mask:
                    copy_nt_.append(
                        (
                            neg_huge,
                            item[..., None, None],
                        )
                    )
                copy_nt = copy_nt_

        params = {
            "term": terms,
            "root": roots,
            "left": rule_left,
            "right": rule_right,
            "head": rule_head,
            "slr": rule_slr,
            "copy_nt": copy_nt,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def get_rules_mask1(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return 1 for not masked
        nt = nt_num_nodes
        pt = pt_num_nodes
        nt_node_mask = torch.ones(
            batch_size, nt_num_nodes, nt_num_nodes, dtype=torch.bool
        )
        pt_node_mask = torch.ones(
            batch_size, nt_num_nodes, pt_num_nodes, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j] = False
                for j, child_span in enumerate(pt_span):
                    if not (is_parent(parent_span, child_span)):
                        pt_node_mask[b, i, j] = False

        node_mask = torch.cat([nt_node_mask, pt_node_mask], dim=2).to(device)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        nt = nt_num_nodes
        node_mask = torch.zeros(batch_size, nt, nt, nt, device=device, dtype=torch.bool)

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

        for b, nt_span in enumerate(nt_spans):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    node_mask[b, i, i, i].fill_(True)
                    for j, child in enumerate(nt_span):
                        if is_strict_parent(parent, child):
                            node_mask[b, i, i, j].fill_(True)
                            node_mask[b, i, j, i].fill_(True)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            node_mask[b, i, j, k].fill_(True)
                            node_mask[b, i, k, j].fill_(True)

        return node_mask.contiguous().view(batch_size, nt, nt, nt)

    def get_rules_mask3(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # mask1 + children

        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        # return True for not masked
        nt = nt_num_nodes
        nt_node_mask = torch.ones(
            batch_size, nt_num_nodes, nt_num_nodes, dtype=torch.bool
        )

        def is_parent(parent, child):
            return child[0] >= parent[0] and child[1] <= parent[1]

        for b, nt_span in enumerate(nt_spans):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j] = False

        for b, nt_spans_inst in enumerate(nt_spans):
            spans, parents, mapping = spans2tree(nt_spans_inst, return_mapping=True)
            for i, span1 in enumerate(spans):
                for j, span2 in enumerate(spans[i + 1 :], start=i + 1):
                    if not (is_parent(span1, span2)):
                        continue
                    depth = 1
                    k = parents[j]
                    while k != -1:
                        if k == i:
                            break
                        k = parents[k]
                        depth += 1
                    if depth > 2:
                        nt_node_mask[b, mapping[i], mapping[j]] = False

        # for mask, spans_inst in zip(nt_node_mask, nt_spans):
        #     self.show_constraint(mask, spans_inst)

        node_mask = nt_node_mask.unsqueeze(3) * nt_node_mask.unsqueeze(2)
        return node_mask.view(batch_size, nt, nt, nt)

    def show_constraint(self, mask, spans):
        # mask should be N * N. not allow batch.
        position = mask[: len(spans), : len(spans)].nonzero(as_tuple=True)
        allowed = defaultdict(list)
        for i, j in zip(*position):
            allowed[i].append(j)
        for i, vset in allowed.items():
            print("Parent: ", spans[i])
            print("Possible children: ", [spans[j] for j in vset])
            print("===")
