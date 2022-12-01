import logging
import operator
from typing import Any, List, Optional

from src.models.general_seq2seq import GeneralSeq2SeqModule
from src.utils.fn import annotate_snt_with_brackets, report_ids_when_err

log = logging.getLogger(__file__)


class GeneralSeq2SeqWithFixedSrcParserModule(GeneralSeq2SeqModule):
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        observed = {
            "x": batch["tgt_ids"],
            "lengths": batch["tgt_lens"],
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
        }
        logging_vals = {}
        src_spans = self.parser.get_spans(batch)

        with self.profiler.profile("src_encoding"):
            x = self.encode(batch)
            node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred = self.decoder(node_features, node_spans)
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            tgt_nll = tgt_pred.dist.nll
        soft_constraint_loss = 0

        tgt_entropy_reg = 0
        if self.training:
            if self.decoder.rule_soft_constraint_solver is not None:
                with self.profiler.profile("compute_soft_constraint"):
                    soft_constraint_loss = self.decoder.get_soft_constraint_loss(tgt_pred)
            if (e := self.hparams.decoder_entropy_reg) > 0:
                entropy = tgt_pred.dist.entropy
                tgt_entropy_reg = -e * entropy
                logging_vals["tgt_entropy"] = entropy
        return {
            "decoder": self.threshold(tgt_nll) + soft_constraint_loss + tgt_entropy_reg,
            "tgt_nll": tgt_nll,
            "runtime": {"seq_encoded": x},
            "tgt_runtime": {"pred": tgt_pred},
            "log": logging_vals,
        }

    @report_ids_when_err
    def forward_visualize(self, batch, sample=False):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        observed = {
            "x": batch["tgt_ids"],
            "lengths": batch["tgt_lens"],
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
        }

        src_spans = self.parser.get_spans(batch)
        src_annotated = []
        for src, spans in zip(batch["src"], src_spans):
            annotated = annotate_snt_with_brackets(src, spans)
            src_annotated.append(annotated)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_pred = self.decoder(node_features, node_spans)
        tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(tgt_pred)

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
                        if copied_span[0] <= tgt_span[0] and tgt_span[1] <= copied_span[1]:
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    if tgt_span[0] == tgt_span[1]:
                        is_copy = tgt_span[2] // num_pt_spans == self.decoder.pt_states - 1
                    elif batch.get("copy_nt") is not None:
                        is_copy = tgt_span[2] // num_nt_spans == self.decoder.nt_states - 1
                    if is_copy:
                        copied.append(tgt_span)
                alignments_inst.append(
                    (
                        " ".join(src_snt[src_span[0] : src_span[1] + 1]) + f" ({src_span[0]}, {src_span[1] + 1})",
                        " ".join(tgt_snt[tgt_span[0] : tgt_span[1] + 1]) + f" ({tgt_span[0]}, {tgt_span[1] + 1})",
                        "COPY" if is_copy else "",
                    )
                )
            alignments.append(alignments_inst[::-1])
        return {
            "src_tree": src_annotated,
            "tgt_tree": tgt_annotated,
            "alignment": alignments,
        }

    @report_ids_when_err
    def forward_generate(self, batch):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]

        src_spans = self.parser.get_spans(batch)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_pred = self.decoder(node_features, node_spans)
        tgt_pred = self.decoder.prepare_sampler(tgt_pred, batch["src"], src_ids)
        y_preds = self.decoder.generate(tgt_pred)

        # import numpy as np
        # tgt_pred = self.decoder.observe_x(batch["tgt_ids"], batch["tgt_lens"], inplace=False)
        # tgt_ppl = np.exp(tgt_pred.dist.nll.detach().cpu().numpy() / np.array(batch["tgt_lens"]))

        return {"pred": [item[0] for item in y_preds]}

    def training_step(self, batch: Any, batch_idx: int, *, forward_prediction=None):
        output = forward_prediction if forward_prediction is not None else self(batch)
        loss = output["decoder"].mean()
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        if "log" in output:
            self.log_dict({"train/" + k: v.mean() for k, v in output["log"].items()})

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"].mean()
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])

        if (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0:
            self.test_step(batch, batch_idx=None)

        if batch_idx == 0:
            single_inst = {key: (value[:2] if key != "transformer_inputs" else value) for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(trees["src_tree"], trees["tgt_tree"], trees["alignment"]):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print("Alg:\n" + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg)))

        return {"loss": loss}
