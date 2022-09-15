import logging
import operator
from typing import Any, List, Optional

import torch
from hydra.utils import instantiate

from src.models.node_filter.base import NodeFilterBase
from src.utils.fn import annotate_snt_with_brackets, report_ids_when_err

from .gseq2seq_fixedsrctree import GeneralSeq2SeqWithFixedSrcParserModule

log = logging.getLogger(__file__)


class GSeq2SeqL0WithFixedSrcParserModule(GeneralSeq2SeqWithFixedSrcParserModule):
    """PR. when decoding, constraint is NOT applied."""

    def __init__(self, node_filter=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

    def setup_patch(self, stage=None, datamodule=None) -> None:
        self.node_filter: NodeFilterBase = instantiate(
            self.hparams.node_filter, dim=self.tree_encoder.get_output_dim()
        )

    @report_ids_when_err
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))
        src_spans = self.parser.get_spans(batch)

        x = self.encode(batch)
        node_features_all, node_spans_all = self.tree_encoder(
            x, src_lens, spans=src_spans
        )

        gates = self.node_filter(node_spans_all, node_features_all, x)
        nf_samples, nf_logprob = self.node_filter.sample(gates)
        filtered_spans = self.node_filter.apply_filter(
            nf_samples, node_spans_all, node_features_all
        )
        src_logprob = nf_logprob
        src_nll = -nf_logprob

        tgt_nll = self.decoder(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features_all,
            node_spans_all,
            copy_position=copy_position,
            filtered_spans=filtered_spans,
        )

        alpha = 0.1
        with torch.no_grad():
            # nf_samples_argmax, nf_logprob_argmax = self.node_filter.argmax(gates)
            # filtered_spans_argmax = self.node_filter.apply_filter(
            #     nf_samples_argmax, node_spans_all, node_features_all
            # )
            # tgt_nll_argmax = self.decoder(
            #     batch["tgt_ids"],
            #     batch["tgt_lens"],
            #     node_features_all,
            #     node_spans_all,
            #     copy_position=copy_position,
            #     filtered_spans=filtered_spans_argmax,
            # )
            num_samples = torch.stack([item.sum() for item in nf_samples])
            # num_samples_argmax = torch.stack([item.sum() for item in nf_samples_argmax])
            neg_reward = tgt_nll.detach() + alpha * num_samples
            # neg_reward = (tgt_nll - tgt_nll_argmax).detach() + alpha * (
            #     num_samples - num_samples_argmax
            # )

        if "copy_phrase" in batch:
            copy_nt = batch["copy_phrase"]

        return {
            "decoder": tgt_nll.mean(),
            "tgt_nll": tgt_nll.mean(),
            "encoder": (src_logprob * neg_reward).mean(),
            "src_nll": src_nll.mean(),
        }

    def forward_visualize(self, batch, sample=False):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        src_spans = self.parser.get_spans(batch)
        src_annotated = []
        for src, spans in zip(batch["src"], src_spans):
            annotated = annotate_snt_with_brackets(src, spans)
            src_annotated.append(annotated)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        gates = self.node_filter(node_spans, node_features, x)
        nf_samples, nf_logprob = self.node_filter.argmax(gates)
        filtered_spans = self.node_filter.apply_filter(
            nf_samples, node_spans, node_features
        )

        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            copy_position=copy_position,
            filtered_spans=filtered_spans,
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

        src_spans = self.parser.get_spans(batch)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        gates = self.node_filter(node_spans, node_features, x)
        nf_samples, nf_logprob = self.node_filter.argmax(gates)
        filtered_spans = self.node_filter.apply_filter(
            nf_samples, node_spans, node_features
        )

        y_preds = self.decoder.generate(
            node_features,
            node_spans,
            self.datamodule.vocab_pair,
            batch["src_ids"],
            batch["src"],
            filtered_spans=filtered_spans,
        )

        return {"pred": [item[0] for item in y_preds]}

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/decoder", output["decoder"], prog_bar=True)
        self.log("train/encoder", output["encoder"], prog_bar=True)

        if batch_idx == 0:
            self.eval()
            single_inst = {key: value[:2] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(
                trees["src_tree"], trees["tgt_tree"], trees["alignment"]
            ):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print(
                    "Alg:\n"
                    + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg))
                )
            self.train()
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])
        return {"loss": loss}
