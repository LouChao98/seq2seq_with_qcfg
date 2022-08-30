import copy
import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...datamodules.components.vocab import VocabularyPair
from ..components.common import MultiResidualLayer
from .base import TgtParserBase
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.pcfg import PCFG, TokenType

log = logging.getLogger(__file__)


class NeuralQCFGPenmanParser(NeuralQCFGTgtParser):
    """A prototype tgt parser for penman outputs"""

    def __init__(self, num_state_adding_bracket, **kwargs):
        super(NeuralQCFGPenmanParser, self).__init__(**kwargs)
        self.num_state_adding_bracket = num_state_adding_bracket

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
                        if len(expanded) >= max_len:
                            break
                        if t == TokenType.VOCAB:
                            expanded.append(v)
                        elif t == TokenType.COPY_PT:
                            span = pt_spans_inst[v]
                            tokens = vocab_pair.src2tgt(
                                src_ids_inst[span[0] : span[1] + 1]
                            )
                            copy_pt[span[0], len(expanded)] = True
                            if tokens[0] == vocab_pair.tgt.unk_token_id:
                                copy_unk[len(expanded)] = span[0]
                            expanded.extend(tokens)
                        elif t == TokenType.COPY_NT:
                            span = nt_spans_inst[v]
                            tokens = vocab_pair.src2tgt(
                                src_ids_inst[span[0] : span[1] + 1]
                            )
                            copy_nt[span[1] - span[0] - 1].append(
                                (span[0], len(expanded))
                            )  # copy_nt starts from w=2
                            for i, token in enumerate(tokens):
                                if token == vocab_pair.tgt.unk_token_id:
                                    copy_unk[len(expanded) + i] = span[0] + i
                            expanded.extend(tokens)

                    if max(expanded) >= len(vocab_pair.tgt):
                        print(111)
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
                            _copy,
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
        copy_position=None,
        observed_position=None,
    ):
        """Generate parameters used in PCFG

        :param node_features: features of spans on the src tree
        :param spans: list of span boundries
        :param x: the target sequence, defaults to None
        :param copy_position: (pt, nt), 1 for can copy, defaults to None
        :param observed_position: pt and nt, pt is just used for add_bracket
            nt should be a list of masks. For observed spans, they will be in
            add_bracket, this is handled here. For impossible spans, nt should mask
            them out, defaults to None
        :return: _description_
        """

        batch_size = len(spans)

        # get params without x
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = super().get_params(
            node_features, spans
        )

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

        params = {"term": terms, "root": roots, "rule": rules, "copy_nt": copy_nt}
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes
