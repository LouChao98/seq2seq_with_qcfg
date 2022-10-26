import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_memlab import profile_every
from torch_scatter import scatter_mean
from torchmetrics import Metric, MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.models.constraint.fully_parameterized import FPSimpleHierarchy
from src.models.src_parser.base import SrcParserBase
from src.models.tgt_parser.base import TgtParserBase
from src.models.tree_encoder.base import TreeEncoderBase
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    extract_parses_span_only,
    get_actions,
    get_tree,
    report_ids_when_err,
)
from src.utils.metric import PerplexityMetric

log = logging.getLogger(__file__)


class GeneralSeq2SeqModule(ModelBase):
    """A module for general seq2seq tasks.

    * support pretrained models
    * encoders
    * custom test metric
    """

    def __init__(
        self,
        embedding=None,
        transformer_pretrained_model=None,
        encoder=None,
        tree_encoder=None,
        decoder=None,
        parser=None,
        optimizer=None,
        scheduler=None,
        test_metric=None,
        load_from_checkpoint=None,
        parser_entropy_reg=0.0,
        decoder_entropy_reg=0.0,
        param_initializer="xavier_uniform",
        track_param_norm=False,
        real_val_every_n_epochs=5,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)

        self.profiler = self.trainer.profiler or PassThroughProfiler()
        if not isinstance(self.profiler, PassThroughProfiler):
            log.warning("Profiler is enabled.")

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        self.embedding = instantiate(
            self.hparams.embedding,
            num_embeddings=len(self.datamodule.src_vocab),
        )
        self.pretrained = (
            AutoModel.from_pretrained(self.hparams.transformer_pretrained_model)
            if self.hparams.transformer_pretrained_model is not None
            else None
        )
        self.encoder = instantiate(
            self.hparams.encoder,
            input_dim=0
            + (0 if self.embedding is None else self.embedding.weight.shape[1])
            + (0 if self.pretrained is None else self.pretrained.config.hidden_size),
        )

        self.parser: SrcParserBase = instantiate(self.hparams.parser, vocab=len(self.datamodule.src_vocab))
        self.tree_encoder: TreeEncoderBase = instantiate(self.hparams.tree_encoder, dim=self.encoder.get_output_dim())
        self.decoder: TgtParserBase = instantiate(
            self.hparams.decoder,
            vocab=len(self.datamodule.tgt_vocab),
            datamodule=self.datamodule,
            src_dim=self.tree_encoder.get_output_dim(),
        )
        self.threshold = torch.nn.Threshold(0.1, 0)

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric: Metric = instantiate(self.hparams.test_metric)

        self.setup_patch(stage, datamodule)

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(self.hparams.load_from_checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        else:
            init_func = {
                "xavier_uniform": nn.init.xavier_uniform_,
                "xavier_normal": nn.init.xavier_normal_,
                "kaiming_uniform": nn.init.kaiming_uniform_,
                "kaiming_normal": nn.init.kaiming_normal_,
            }
            init_func = init_func[self.hparams.param_initializer]
            for name, param in self.named_parameters():
                if param.dim() > 1:
                    init_func(param)
                elif "norm" not in name:
                    nn.init.zeros_(param)

    def setup_patch(self, stage: Optional[str] = None, datamodule=None):
        # allow submodule changing submodules before loading checkpoint
        ...

    def encode(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = []
        if self.embedding is not None:
            h = self.embedding(src_ids)
            x.append(h)
        if self.pretrained is not None:
            h = self.pretrained(**batch["transformer_inputs"])[0]
            if len(h) > len(src_ids):
                h = h[: len(src_ids)]
            out = torch.zeros(
                src_ids.shape[0],
                src_ids.shape[1] + 1,
                h.shape[-1],
                device=src_ids.device,
            )
            scatter_mean(h, batch["transformer_offset"], 1, out=out)
            out = out[:, 1:]
            x.append(out)
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]
        x = self.encoder(x, src_lens)
        return x

    @report_ids_when_err
    @profile_every(5, enable=False)
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        observed = {
            "x": batch["tgt_ids"],
            "lengths": batch["tgt_lens"],
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
            "prior_alignment": batch.get("prior_alignment"),
        }
        logging_vals = {}

        with self.profiler.profile("compute_src_nll_and_sampling"):
            dist = self.parser(src_ids, src_lens)
            src_nll = -dist.partition
            src_event, src_logprob = self.parser.sample(src_ids, src_lens, dist=dist)
            src_spans = extract_parses_span_only(src_event[-1], src_lens, inc=1)

        with self.profiler.profile("src_encoding"):
            x = self.encode(batch)
            node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred = self.decoder(node_features, node_spans)
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            tgt_nll = tgt_pred.dist.nll

        with self.profiler.profile("reward"), torch.no_grad():
            src_event_argmax, src_logprob_argmax = self.parser.argmax(src_ids, src_lens, dist=dist)
            src_spans_argmax = extract_parses_span_only(src_event_argmax[-1], src_lens, inc=1)
            node_features_argmax, node_spans_argmax = self.tree_encoder(x, src_lens, spans=src_spans_argmax)
            tgt_argmax_pred = self.decoder(node_features_argmax, node_spans_argmax)
            tgt_argmax_pred = self.decoder.observe_x(tgt_argmax_pred, **observed)
            tgt_nll_argmax = tgt_argmax_pred.dist.nll
            neg_reward = (tgt_nll - tgt_nll_argmax).detach()
            logging_vals["reward"] = -neg_reward

        objective = src_logprob * neg_reward

        soft_constraint_loss = 0
        src_entropy_reg = 0
        tgt_entropy_reg = 0
        tgt_loss = tgt_nll
        if self.training:
            if self.decoder.rule_soft_constraint_solver is not None:
                reward = self.decoder.rule_soft_constraint.get_reward_from_pred(tgt_pred)
                mask = FPSimpleHierarchy().get_mask_from_pred(tgt_pred)
                dist = tgt_pred.dist.spawn(
                    params={
                        "term": torch.where(tgt_pred.dist.params["term"] > -1e8, 0, -1e9),
                        "rule": torch.where(tgt_pred.dist.params["rule"] > -1e8, (reward + 1e-9).log(), -1e9),
                        "root": torch.where(tgt_pred.dist.params["root"] > -1e8, 0.0, -1e9),
                    }
                )
                # rule = tgt_pred.dist.params['rule'].clone()
                # rule[~mask] = -1e9
                # dist2 =tgt_pred.dist.spawn(params={'rule': rule, 'constraint': None})
                tgt_dist = tgt_pred.dist.spawn(params={"constraint": None})
                tgt_loss = dist.cross_entropy(tgt_dist, fix_left=True)
            if (e := self.hparams.parser_entropy_reg) > 0:
                entropy = self.parser.entropy(src_ids, src_lens, dist)
                src_entropy_reg = -e * entropy
                logging_vals["src_entropy"] = entropy
            if (e := self.hparams.decoder_entropy_reg) > 0:
                entropy = tgt_pred.dist.entropy
                tgt_entropy_reg = -e * entropy
                logging_vals["tgt_entropy"] = entropy

        return {
            "decoder": tgt_loss + soft_constraint_loss + tgt_entropy_reg,
            "encoder": self.threshold(src_nll) + objective + src_entropy_reg,
            "tgt_nll": tgt_nll,
            "src_nll": src_nll,
            "runtime": {"seq_encoded": x},
            "src_runtime": {
                "dist": dist,
                "event": src_event,
                "argmax_event": src_event_argmax,
            },
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
            "prior_alignment": batch.get("prior_alignment"),
        }

        parse = self.parser.sample if sample else self.parser.argmax
        src_spans = parse(src_ids, src_lens)[0][-1]
        src_spans, src_trees = extract_parses(src_spans, src_lens, inc=1)
        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))

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
                if src_span is None:
                    continue
                is_copy = False
                if getattr(self.decoder, "use_copy"):
                    should_skip = False
                    for copied_span in copied:
                        if copied_span[0] <= tgt_span[0] and tgt_span[1] <= copied_span[1]:
                            should_skip = True
                            break
                    if should_skip:
                        continue
                    # if isinstance(tgt_span[2],  str):
                    if tgt_span[2] == "p":
                        is_copy = tgt_span[3] == self.decoder.pt_states - 1
                    elif batch.get("copy_phrase") is not None:
                        is_copy = tgt_span[3] == self.decoder.nt_states - 1
                    # else:
                    #     if tgt_span[0] == tgt_span[1]:
                    #         is_copy = tgt_span[2] // num_pt_spans == self.decoder.pt_states - 1
                    #     elif batch.get("copy_nt") is not None:
                    #         is_copy = tgt_span[2] // num_nt_spans == self.decoder.nt_states - 1
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

        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist=dist)[0]
        src_spans = extract_parses_span_only(src_spans[-1], src_lens, inc=1)

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
        loss_decoder = output["decoder"].mean()
        loss_encoder = output["encoder"].mean()
        loss = loss_encoder + loss_decoder
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/src", loss_encoder, prog_bar=True)
        self.log("train/tgt", loss_decoder, prog_bar=True)
        if "log" in output:
            self.log_dict({"train/" + k: v.mean() for k, v in output["log"].items()})
        if batch_idx == 0:
            self.eval()
            single_inst = {key: (value[:2] if key != "transformer_inputs" else value) for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(trees["src_tree"], trees["tgt_tree"], trees["alignment"]):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print("Alg:\n" + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg)))
            self.train()
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_metric.reset()
        self.test_metric.reset()
        if self.hparams.track_param_norm:
            self.log(
                "stat/parser_norm",
                sum(param.norm().item() ** 2 for param in self.parser.parameters()) ** 0.5,
            )
            self.log(
                "stat/encoder_norm",
                sum(param.norm().item() ** 2 for param in self.tree_encoder.parameters()) ** 0.5,
            )
            self.log(
                "stat/decoder_norm",
                sum(param.norm().item() ** 2 for param in self.decoder.parameters()) ** 0.5,
            )

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss_decoder = output["decoder"].mean()
        loss_encoder = output["encoder"].mean()
        loss = loss_encoder + loss_decoder
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

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute()
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)
        self.print("val/epoch", str(self.current_epoch + 1))
        self.print("val/ppl", str(ppl.item()))
        if (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0:
            acc = self.test_metric.compute()
            self.log_dict({"val/" + k: v for k, v in acc.items()})
            self.print(acc)

    def on_test_epoch_start(self) -> None:
        self.test_metric.reset()
        return super().on_test_epoch_start()

    @torch.inference_mode(False)
    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        # patch for inference_mode
        batch["src_ids"] = batch["src_ids"].clone()
        batch["tgt_ids"] = batch["tgt_ids"].clone()

        preds = self.forward_generate(batch)["pred"]
        targets = batch["tgt"]

        self.test_metric(preds, targets)

        # if batch_idx == 0:
        #     single_inst = {key: value[:2] for key, value in batch.items()}
        #     trees = self.forward_visualize(single_inst)
        #     self.print("=" * 79)
        #     for src, tgt, alg in zip(
        #         trees["src_tree"], trees["tgt_tree"], trees["alignment"]
        #     ):
        #         self.print("Src:", src)
        #         self.print("Tgt:", tgt)
        #         self.print(
        #             "Alg:\n"
        #             + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg))
        #         )

        return {"preds": preds, "targets": targets, "id": batch["id"]}

    def test_epoch_end(self, outputs) -> None:
        acc = self.test_metric.compute()
        d = {"test/" + k: v for k, v in acc.items()}
        d["step"] = self.global_step
        self.log_dict(d)
        self.print(acc)
        self.save_predictions(outputs)
