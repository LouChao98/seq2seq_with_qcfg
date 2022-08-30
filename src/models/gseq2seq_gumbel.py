import logging
import operator
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import torch

from src.models.general_seq2seq import GeneralSeq2SeqModule
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    get_actions,
    get_tree,
)

from .components.span import CombinarySpanEncoder

log = logging.getLogger(__file__)


class GSeq2seqGumbel(GeneralSeq2SeqModule):
    """Based on GeneralSeq2Seq
    instead of REINFORCE, here we use gumbel-max tricks
    """

    def __init__(self, *args, span_encoder_dropout=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

    def setup_patch(self, stage: Optional[str] = None, datamodule=None) -> None:
        self.span_encoder = CombinarySpanEncoder(
            in_dim=self.encoder.get_output_dim(),
            out_dim=self.encoder.get_output_dim(),
            dropout=self.hparams.span_encoder_dropout,
        )

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        x = self.encode(batch)

        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition
        event, src_logprob, trace, buffer = self.parser.gumbel_sample(
            src_ids, src_lens, dist
        )
        src_spans = extract_parses(event[-1], src_lens, inc=1)[0]

        spans, node_feats = self.collect_node_feature(src_spans, trace, buffer, x)
        node_feats, node_spans = self.tree_encoder(node_feats, src_lens, spans=spans)

        tgt_nll = self.decoder(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_feats,
            node_spans,
            copy_position=copy_position,
        )

        del buffer

        return {
            "decoder": tgt_nll.mean(),
            "encoder": src_nll.mean(),
            "tgt_nll": tgt_nll.mean(),
            "src_nll": src_nll.mean(),
        }

    def forward_visualize(self, batch, sample=False):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        x = self.encode(batch)

        parse = self.parser.sample if sample else self.parser.argmax
        event, src_logprob = parse(src_ids, src_lens)
        src_spans, src_trees = extract_parses(event[-1], src_lens, inc=1)
        event = event[-1].sum(-1)

        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))

        spans, node_feats = self.collect_node_feature2(src_spans, event, x)
        node_feats, node_spans = self.tree_encoder(node_feats, src_lens, spans=spans)

        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_feats,
            node_spans,
            copy_position=copy_position,
        )
        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)

        num_pt_spans = max(len(item) for item in pt_spans)
        num_nt_spans = max(len(item) for item in nt_spans)

        alignments = []
        for (
            tgt_spans_inst,
            tgt_snt,
            aligned_spans_inst,
            src_snt,
        ) in zip(tgt_spans, batch["tgt"], aligned_spans, batch["src"]):
            alignments_inst = []
            # large span first for handling copy.
            idx = list(range(len(tgt_spans_inst)))
            idx.sort(key=lambda i: (operator.sub(*tgt_spans_inst[i][:2]), -i))
            copied = []
            # for tgt_span, src_span in zip(tgt_spans_inst, aligned_spans_inst):
            for i in idx:
                tgt_span = tgt_spans_inst[i]
                src_span = aligned_spans_inst[i]
                is_copy = False
                if getattr(self.decoder, "use_copy"):
                    should_skip = False
                    for copied_span in copied:
                        if (
                            copied_span[0] <= tgt_span[0]
                            and tgt_span[1] <= copied_span[1]
                        ):
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    if tgt_span[0] == tgt_span[1]:
                        is_copy = (
                            tgt_span[2] // num_pt_spans == self.decoder.pt_states - 1
                        )
                    else:
                        is_copy = (
                            tgt_span[2] // num_nt_spans == self.decoder.nt_states - 1
                        )
                    if is_copy:
                        copied.append(tgt_span)
                alignments_inst.append(
                    (
                        " ".join(src_snt[src_span[0] : src_span[1] + 1])
                        + f" ({src_span[0]}, {src_span[1]+1})",
                        " ".join(tgt_snt[tgt_span[0] : tgt_span[1] + 1])
                        + f" ({tgt_span[0]}, {tgt_span[1]+1})",
                        "COPY" if is_copy else "",
                    )
                )
            alignments.append(alignments_inst[::-1])
        return {
            "src_tree": src_annotated,
            "tgt_tree": tgt_annotated,
            "alignment": alignments,
        }

    def forward_inference(self, batch):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.encode(batch)

        dist = self.parser(src_ids, src_lens)
        event = self.parser.argmax(src_ids, src_lens, dist)[0]
        src_spans = extract_parses(event[-1], src_lens, inc=1)[0]
        event = event[-1].sum(-1)

        spans, node_feats = self.collect_node_feature2(src_spans, event, x)
        node_feats, node_spans = self.tree_encoder(node_feats, src_lens, spans=spans)

        y_preds = self.decoder.generate(
            node_feats,
            node_spans,
            self.datamodule.vocab_pair,
            batch["src_ids"],
            batch["src"],
        )

        # tgt_nll = self.decoder(
        #     batch["tgt_ids"],
        #     batch["tgt_lens"],
        #     node_features,
        #     node_spans,
        #     (batch.get('copy_token'), batch.get('copy_phrase')),
        # )
        # tgt_ppl = np.exp(tgt_nll.detach().cpu().numpy() / batch["tgt_lens"])

        return {"pred": [item[0] for item in y_preds]}

    def collect_node_feature(self, src_spans, trace, buffer, x):
        invtrace = {v: k for k, v in trace.items()}

        node_feats = []
        spans = []

        processed_buffer = {}
        for i, item in buffer.items():
            t = trace[i][1]
            if t == "1234":
                processed_buffer[i] = item[0, :, :, 0]  # <b, start, choice>
            elif t == "YZ":
                processed_buffer[i] = item[0, :, :, 0, 0]  # <b, start, split>
        buffer = processed_buffer

        for b, spans_inst in enumerate(src_spans):
            features_inst = []
            seq_len = len(spans_inst) + 1
            buffer_inst = {i: item[b] for i, item in buffer.items()}  # <start, width>
            spans_inst = [(i, i, None) for i in range(seq_len)] + [
                (s, e, None) for s, e, l in spans_inst
            ]

            # s == e
            single_x = x[b, :seq_len] * 2
            features_inst.extend(list(self.span_encoder(single_x, single_x)))
            for s, e, _ in spans_inst:
                w = e - s
                if w == 0:  # TODO revisit scoring
                    continue
                    # features_inst.append(self.span_encoder(2 * x[b, s], 2 * x[b, e]))
                elif w == 1:
                    features_inst.append(self.span_encoder(single_x[s], single_x[e]))
                else:
                    left = x[b, s, None]
                    right = x[b, e, None]
                    mid1 = x[b, s:e]
                    mid2 = x[b, s + 1 : e + 1]

                    span = self.span_encoder(left + mid1, mid2 + right)

                    # XYZ XYz XyZ Xyz
                    switch = buffer_inst[invtrace[(w, "1234")]][s]
                    yz = buffer_inst[invtrace[(w, "YZ")]][s]
                    d = yz * switch[0]
                    d[0] += switch[2]
                    d[-1] += switch[1]
                    # assert d.sum().item() == 1

                    feat = d @ span
                    features_inst.append(feat)
            node_feats.append(features_inst)
            spans.append(spans_inst)
        return spans, node_feats

    def collect_node_feature2(self, src_spans, event, x):
        event = torch.cat(
            [torch.ones(event.shape[0], 1, event.shape[2], device=event.device), event],
            dim=1,
        )

        node_feats = []
        spans = []
        for b, spans_inst in enumerate(src_spans):
            features_inst = []
            seq_len = len(spans_inst) + 1
            event_inst = event[b]
            spans_inst = [(i, i, None) for i in range(seq_len)] + [
                (s, e, None) for s, e, l in spans_inst
            ]

            # s == e
            single_x = x[b, :seq_len] * 2
            features_inst.extend(list(self.span_encoder(single_x, single_x)))
            for s, e, _ in spans_inst:
                if s == e:  # TODO revisit scoring
                    continue
                    # features_inst.append(self.span_encoder(2 * x[b, s], 2 * x[b, e]))
                elif e - s <= 1:
                    features_inst.append(self.span_encoder(single_x[s], single_x[e]))
                else:
                    left = x[b, s, None]
                    right = x[b, e, None]
                    mid1 = x[b, s:e]
                    mid2 = x[b, s + 1 : e + 1]

                    span = self.span_encoder(left + mid1, mid2 + right)
                    subl = event_inst[: e - s, s]
                    subr = antidiagonal_stride(event_inst, (0, e), e - s)
                    d = subl * torch.flip(subr, (0,))
                    # assert d.sum().item() == 1
                    feat = d @ span
                    features_inst.append(feat)
            node_feats.append(features_inst)
            spans.append(spans_inst)
        return spans, node_feats


def antidiagonal_stride(a: torch.Tensor, offset, size):
    assert a.is_contiguous()
    stride = list(a.stride())
    new_stride = [stride[0] - stride[1]]
    # if len(a.shape) > 3:
    #     new_stride.extend(stride[3:])
    #     return a.as_strided(
    #         size=(a.shape[0], size, *list(a.shape[3:])),
    #         stride=new_stride,
    #         storage_offset=offset[0] * stride[1] + offset[1] * stride[2]
    #     )
    # else:
    return a.as_strided(
        size=(size,),
        stride=new_stride,
        storage_offset=offset[0] * stride[0]
        + offset[1] * stride[1]
        + a.storage_offset(),
    )
