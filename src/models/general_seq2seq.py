import logging
import operator
from functools import partial
from io import StringIO
from itertools import chain
from random import random
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_memlab import profile_every
from torch.autograd import grad
from torch_scatter import scatter_mean
from torchmetrics import Metric, MinMetric
from transformers import AutoModel
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding

import wandb
from src.models.base import ModelBase
from src.models.posterior_regularization.general import NeqNT, NeqPT, NoManyToOneNT, NoManyToOnePT
from src.models.posterior_regularization.pr import MultiTask, compute_pr
from src.models.src_parser.base import SrcParserBase
from src.models.struct.semiring import LogSemiring
from src.models.tgt_parser.base import NO_COPY_SPAN, TgtParserBase
from src.models.tree_encoder.base import TreeEncoderBase
from src.utils.fn import annotate_snt_with_brackets, apply_to_nested_tensor, report_ids_when_err
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
        fix_pretrained=True,
        encoder=None,
        tree_encoder=None,
        decoder=None,
        parser=None,
        optimizer=None,
        scheduler=None,
        test_metric=None,
        load_from_checkpoint=None,
        load_src_parser_from_checkpoint=None,
        load_tgt_parser_from_checkpoint=None,
        param_initializer="xavier_uniform",
        real_val_every_n_epochs=5,
        visualize_every_n_steps=0,
        export_detailed_prediction=True,
        # extension
        warmup_pcfg=0,
        warmup_qcfg=0,
        tgt_annealing=None,
        length_calibrate=False,
        pr_pt_neq_reg=0.0,
        pr_nt_neq_reg=0.0,
        pr_neq_impl=1,
        pr_args=None,
        noisy_spans_reg=0.0,
        noisy_spans_num=0.0,
        parser_entropy_reg=0.0,
        decoder_entropy_reg=0.0,
        soft_constraint_loss_pr=0.0,
        soft_constraint_loss_rl=False,
        soft_constraint_loss_raml=False,
        mini_target=False,
        mini_pcfg=False,
        positional_emb_before_tree_enc=False,
    ):
        assert warmup_pcfg <= warmup_qcfg
        self.warmup = warmup_pcfg
        super().__init__()
        if tgt_annealing is not None:
            log.warning("Tgt annealing is activated.")
            self.add_dynamic_cfg("|tgt_annealing", tgt_annealing)
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
        if self.pretrained is not None and self.hparams.fix_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad_(False)

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

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric: Metric = instantiate(self.hparams.test_metric)

        if self.hparams.mini_pcfg:
            self.dummy_node_emb = nn.Parameter(torch.randn(3, self.tree_encoder.get_output_dim()))

        if self.datamodule.hparams.get("emphasize"):
            assert self.embedding is not None
            self.emphasize_emb = nn.Embedding(3, self.embedding.weight.shape[1])

        if self.hparams.positional_emb_before_tree_enc:
            self.position_emb = RoFormerSinusoidalPositionalEmbedding(
                datamodule.hparams.get("max_src_len", 60), self.encoder.get_output_dim()
            )

        self.setup_patch(stage, self.datamodule)

        if wandb.run is not None:
            tags = []
            for module in [self.encoder, self.tree_encoder, self.parser, self.decoder, self]:
                tags.append(type(module).__name__)
            if self.embedding is not None:
                tags.append("staticEmb")
            if self.pretrained is not None:
                tags.append(self.pretrained.name_or_path)
            wandb.run.tags = wandb.run.tags + tuple(tags)

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
                if name.startswith("pretrained."):
                    continue
                if param.dim() > 1:
                    init_func(param)
                # elif "norm" not in name.lower():
                #     nn.init.zeros_(param)

        if self.hparams.load_src_parser_from_checkpoint is not None:
            ...

        if self.hparams.load_tgt_parser_from_checkpoint is not None:
            ...

    def setup_patch(self, stage: Optional[str] = None, datamodule=None):
        # allow submodule changing submodules before loading checkpoint
        ...

    def encode(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = []
        if self.embedding is not None:
            h = self.embedding(src_ids)
            if hasattr(self, "emphasize_emb"):
                eh = self.emphasize_emb(batch["emphasize"])
                h = h + eh
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

        if self.hparams.positional_emb_before_tree_enc:
            pos_emb = self.position_emb(x.shape[1]).unsqueeze(0)
            x = x + pos_emb
        return x

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.hparams.tgt_annealing is not None:
            self.decoder.temperature = self.hparams.tgt_annealing
            self.log("train/tgt_annealing", self.hparams.tgt_annealing)

    @report_ids_when_err
    @profile_every(5, enable=False)
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        tgt_ids, tgt_lens = batch["tgt_ids"], batch["tgt_lens"]
        extra_scores = {"observed_mask": batch.get("src_masks")}
        observed = {
            "x": tgt_ids,
            "lengths": tgt_lens,
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
            "pt_align": batch.get("align_token"),
            "observed_mask": batch.get("tgt_masks"),
        }
        logging_vals = {}

        with self.profiler.profile("compute_src_nll_and_sampling"):
            dist = self.parser(src_ids, src_lens, extra_scores=extra_scores)
            src_loss = src_nll = dist.nll
            src_event, src_logprob = self.parser.sample(src_ids, src_lens, dist=dist)
            src_spans = src_event["span"]

        with self.profiler.profile("src_encoding"):
            x = self.encode(batch)
            # x = x.detach().requires_grad_()  # For debug: cross instance reference
            # span labels are discarded (set to -1)
            node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        if self.current_epoch < self.hparams.warmup_pcfg:
            node_features = apply_to_nested_tensor(node_features, lambda x: torch.zeros_like(x))

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred = self.decoder(node_features, node_spans)
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            tgt_loss = tgt_nll = tgt_pred.dist.nll

        if self.current_epoch < self.hparams.warmup_pcfg:
            objective = 0.0
        else:
            with self.profiler.profile("reward"), torch.no_grad():
                src_spans_argmax = self.parser.argmax(src_ids, src_lens, dist=dist)
                node_features_argmax, node_spans_argmax = self.tree_encoder(x, src_lens, spans=src_spans_argmax)
                tgt_argmax_pred = self.decoder(node_features_argmax, node_spans_argmax)
                tgt_argmax_pred = self.decoder.observe_x(tgt_argmax_pred, **observed)
                tgt_nll_argmax = tgt_argmax_pred.dist.nll
                neg_reward = (tgt_nll - tgt_nll_argmax).detach()
                logging_vals["reward"] = -neg_reward

            objective = src_logprob * neg_reward

        soft_constraint_pr_loss = 0
        noisy_span_loss = 0
        pt_prior_loss = 0
        src_entropy_reg = 0
        tgt_entropy_reg = 0
        pr_neq_pt_reg = 0
        pr_neq_nt_reg = 0
        length_calibrate_term = 0
        mini_target = 0
        mini_pcfg = 0

        if self.training and self.current_epoch >= self.hparams.warmup_qcfg:
            if self.hparams.soft_constraint_loss_rl:
                tgt_loss = self.decoder.get_rl_loss(tgt_pred, self.hparams.decoder_entropy_reg)
            if self.hparams.soft_constraint_loss_raml:
                tgt_loss = self.decoder.get_raml_loss(tgt_pred)

            if (e := self.hparams.soft_constraint_loss_pr) > 0:
                soft_constraint_pr_loss = self.decoder.get_soft_constraint_loss(tgt_pred)
                logging_vals["soft_constraint_pr"] = soft_constraint_pr_loss
                soft_constraint_pr_loss = e * soft_constraint_pr_loss

            if (e := self.hparams.parser_entropy_reg) > 0:
                entropy = self.parser.entropy(src_ids, src_lens, dist)
                src_entropy_reg = -e * entropy
                logging_vals["src_entropy"] = entropy
            if (e := self.hparams.decoder_entropy_reg) > 0:
                entropy = tgt_pred.dist.entropy
                tgt_entropy_reg = -e * entropy
                logging_vals["tgt_entropy"] = entropy

            if (e := self.hparams.noisy_spans_reg) > 0:
                l = self.decoder.get_noisy_span_loss(
                    node_features, node_spans, self.hparams.noisy_spans_num, observed
                )
                logging_vals["noisy_spans"] = l
                noisy_span_loss = e * l

            pr_args = {} if self.hparams.pr_args is None else self.hparams.pr_args

            if self.hparams.pr_neq_impl == 1:
                neq_pt, neq_nt = NeqPT, NeqNT
            else:
                neq_pt, neq_nt = NoManyToOnePT, NoManyToOneNT

            if (e1 := self.hparams.pr_pt_neq_reg) > 0 and (e2 := self.hparams.pr_nt_neq_reg) > 0 and e1 == e2:
                l = compute_pr(tgt_pred, (None, None), MultiTask(neq_pt(), neq_nt()), **pr_args)
                logging_vals["pr_neq_pt_nt"] = l
                pr_neq_pt_reg = l * e

            else:
                if (e := self.hparams.pr_pt_neq_reg) > 0:
                    l = compute_pr(tgt_pred, None, neq_pt(), **pr_args)
                    logging_vals["pr_neq_pt"] = l
                    pr_neq_pt_reg = l * e
                if (e := self.hparams.pr_nt_neq_reg) > 0:
                    l = compute_pr(tgt_pred, None, neq_nt(), **pr_args)
                    logging_vals["pr_neq_nt"] = l
                    pr_neq_nt_reg = l * e

            if (prior_alignment := batch.get("prior_alignment")) is not None:
                prior_alignment = prior_alignment.transpose(1, 2)
                logZ, trace = tgt_pred.dist.inside(tgt_pred.dist.params, LogSemiring, use_reentrant=False)
                term_m = grad(logZ.sum(), [tgt_pred.dist.params["term"]], create_graph=True)[0]
                term_m = term_m.view(tgt_pred.batch_size, -1, tgt_pred.pt_states, tgt_pred.pt_num_nodes)
                term_m = term_m.sum(2)[:, : prior_alignment.shape[1], : prior_alignment.shape[2]]
                pt_prior_loss = -(prior_alignment * term_m.clamp(1e-9).log()).sum((1, 2))
                logging_vals["pt_prior_ce"] = pt_prior_loss

            if self.hparams.length_calibrate:
                length_calibrate_term = -tgt_pred.dist.partition_at_length(tgt_pred.params, tgt_lens)

            if self.hparams.mini_target:
                mini_node_features = [item[-1:].expand(2, -1) for item in node_features]
                mini_node_spans = [[(0, 1, NO_COPY_SPAN)] for _ in range(len(node_spans))]
                nt_range, pt_range = self.decoder.nt_span_range, self.decoder.pt_span_range
                self.decoder.nt_span_range = self.decoder.pt_span_range = [1, 1]
                mini_pred = self.decoder(mini_node_features, mini_node_spans)
                mini_pred = self.decoder.observe_x(mini_pred, **observed)
                mini_target = mini_pred.dist.nll
                logging_vals["mini_target"] = mini_target
                self.decoder.nt_span_range, self.decoder.pt_span_range = nt_range, pt_range

            if self.hparams.mini_pcfg:
                dummy_node_features = [self.dummy_node_emb for _ in range(len(src_ids))]
                dummy_spans = [
                    [(0, 1, NO_COPY_SPAN), (1, 2, NO_COPY_SPAN), (0, 2, NO_COPY_SPAN)] for _ in range(len(src_ids))
                ]
                mini_pred = self.decoder(dummy_node_features, dummy_spans)
                mini_pred = self.decoder.observe_x(mini_pred, **observed)
                mini_pcfg = mini_pred.dist.nll
                logging_vals["mini_pcfg"] = mini_pcfg

            if hasattr(tgt_pred, "vq_commit_loss"):
                # TODO refactor this
                tgt_loss = tgt_loss + tgt_pred.vq_commit_loss
                logging_vals["commit_loss"] = tgt_pred.vq_commit_loss

            if hasattr(tgt_pred, "score_reg_loss"):
                tgt_loss = tgt_loss + tgt_pred.score_reg_loss
                logging_vals["score_reg_loss"] = tgt_pred.score_reg_loss

            if hasattr(tgt_pred, "loss"):
                tgt_loss = tgt_loss + tgt_pred.loss

            if hasattr(tgt_pred, "log"):
                logging_vals.update(tgt_pred.log)

        return {
            "decoder": tgt_loss
            + length_calibrate_term
            + soft_constraint_pr_loss
            + tgt_entropy_reg
            + noisy_span_loss
            + pr_neq_pt_reg
            + pr_neq_nt_reg
            + pt_prior_loss
            + mini_target
            + mini_pcfg,
            "encoder": src_loss + objective + src_entropy_reg,
            "tgt_nll": tgt_nll,
            "src_nll": src_nll,
            "runtime": {"seq_encoded": x},
            "src_runtime": {
                "dist": dist,
                "event": src_event,
            },
            "tgt_runtime": {"pred": tgt_pred},
            "log": logging_vals,
        }

    @report_ids_when_err
    def forward_visualize(self, batch):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        extra_scores = {"observed_mask": batch.get("src_masks")}
        observed = {
            "x": batch["tgt_ids"],
            "lengths": batch["tgt_lens"],
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
            "pt_align": batch.get("align_token"),
            # "prior_alignment": batch.get("prior_alignment"),
            # "observed_mask": batch.get("tgt_masks"),
        }

        src_spans = self.parser.argmax(src_ids, src_lens, extra_scores=extra_scores)
        src_annotated = []
        for snt, src_span_inst in zip(batch["src"], src_spans):
            tree = annotate_snt_with_brackets(snt, src_span_inst)
            src_annotated.append(tree)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        if self.current_epoch < self.hparams.warmup_pcfg:
            node_features = apply_to_nested_tensor(node_features, lambda x: torch.zeros_like(x))

        # self.decoder.use_fast = False

        tgt_pred = self.decoder(node_features, node_spans)
        tgt_pred = self.decoder.observe_x(tgt_pred, **observed)

        # # visualize pr constrained dist
        # constraint_feature = self.decoder.rule_soft_constraint.get_feature_from_pred(tgt_pred)
        # dist = self.decoder.rule_soft_constraint_solver(tgt_pred, constraint_feature, get_dist=True)
        # tgt_pred.dist = dist

        # visualize reward: nodecomp
        # reward = self.decoder.rule_soft_constraint.get_weight_from_pred(tgt_pred)
        # dist = tgt_pred.dist.spawn(
        #     params={
        #         "term": torch.where(tgt_pred.dist.params["term"] > -1e8, 0, -1e9),
        #         "rule": torch.where(tgt_pred.dist.params["rule"] > -1e8, reward, -1e9),
        #         "root": torch.where(tgt_pred.dist.params["root"] > -1e8, 0.0, -1e9),
        #     }
        # )
        # tgt_pred.dist = dist

        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(tgt_pred)
        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)

        # self.decoder.use_fast = True

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
                    if isinstance(tgt_span[2], str):
                        if tgt_span[2] == "p":
                            is_copy = tgt_span[3] == self.decoder.pt_states - 1
                        elif batch.get("copy_phrase") is not None:
                            is_copy = tgt_span[3] == self.decoder.nt_states - 1
                    else:
                        if tgt_span[0] == tgt_span[1]:
                            is_copy = tgt_span[2] // num_pt_spans == self.decoder.pt_states - 1
                        elif batch.get("copy_nt") is not None:
                            is_copy = tgt_span[2] // num_nt_spans == self.decoder.nt_states - 1
                    if is_copy:
                        copied.append(tgt_span)
                alignments_inst.append(
                    (
                        " ".join(src_snt[src_span[0] : src_span[1]]) + f" ({src_span[0]}, {src_span[1]})",
                        " ".join(tgt_snt[tgt_span[0] : tgt_span[1]]) + f" ({tgt_span[0]}, {tgt_span[1]})",
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
    def forward_generate(self, batch, get_baseline=False):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]

        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist=dist)

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_pred = self.decoder(node_features, node_spans)
        tgt_pred = self.decoder.prepare_sampler(tgt_pred, batch["src"], src_ids)
        y_preds = self.decoder.generate(tgt_pred)  # , length_hint=batch['tgt_lens'])

        if get_baseline:
            # TODO this always be ppl. but scores_on_predicted can be others
            observed = {
                "x": batch["tgt_ids"],
                "lengths": batch["tgt_lens"],
                "pt_copy": batch.get("copy_token"),
                "nt_copy": batch.get("copy_phrase"),
            }
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            baseline = np.exp(tgt_pred.dist.nll.detach().cpu().numpy() / np.array(batch["tgt_lens"])).tolist()
        else:
            baseline = None

        return {"pred": [item[0] for item in y_preds], "score": [item[1] for item in y_preds], "baseline": baseline}

    def print_prediction(self, batch, handler=None, batch_size=1):
        training_state = self.training
        self.train(False)
        if handler is None:
            handler = self.print
        if isinstance(batch_size, int):
            batch = make_subbatch(batch, batch_size)
        trees = self.forward_visualize(batch)
        for src, tgt, alg in zip(trees["src_tree"], trees["tgt_tree"], trees["alignment"]):
            handler("Src:  ", src)
            handler("Tgt:  ", tgt)
            handler("Alignment:\n" + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg)))
            handler("=" * 79)
        self.train(training_state)

    def training_step(self, batch: Any, batch_idx: int, *, forward_prediction=None):
        # self.print(f'{len(batch["src_lens"])} - {max(batch["src_lens"])} - {max(batch["tgt_lens"])}')

        output = forward_prediction if forward_prediction is not None else self(batch)

        # # For debug: cross instance reference
        # output['decoder'][2].backward()
        # g = output['runtime']['seq_encoded'].grad.flatten(1).sum(1)

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

        if batch_idx == 0 or (
            self.hparams.visualize_every_n_steps > 0
            and (self.trainer.global_step + 1) % self.hparams.visualize_every_n_steps == 0
        ):
            self.print_prediction(batch)

        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_metric.reset()
        self.test_metric.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss_decoder = output["decoder"].mean()
        loss_encoder = output["encoder"].mean()
        loss = loss_encoder + loss_decoder
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])

        if self.trainer.sanity_checking or (
            (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0 and self.current_epoch >= self.warmup
        ):
            predicted = self.test_step(batch, batch_idx=None)
        else:
            predicted = {}

        # if batch_idx == 0:
        #     self.print_prediction(batch)

        return {"loss": loss} | predicted

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute().item()
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)
        self.print("val/epoch", str(self.current_epoch + 1))
        self.print("val/ppl", str(ppl))

        if self.trainer.sanity_checking or (
            (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0 and self.current_epoch > self.warmup
        ):
            acc = self.test_metric.compute()
            if not isinstance(acc, dict):
                acc = {"result": acc}
            self.log_dict({"val/" + k: v for k, v in acc.items()})
            self.print(acc)
            self.save_predictions(outputs, f"predict_on_val_epoch{self.current_epoch}")
            if outputs[0].get("detailed") is not None:
                self.save_detailed_predictions(outputs, f"detailed_predict_on_val_epoch{self.current_epoch}")

    @torch.inference_mode(False)
    def on_test_epoch_start(self) -> None:
        self.test_metric.reset()
        return super().on_test_epoch_start()

    @torch.inference_mode(False)
    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        # patch for inference_mode
        batch = apply_to_nested_tensor(batch, func=lambda x: x.clone())

        preds = self.forward_generate(batch, get_baseline=self.hparams.export_detailed_prediction)
        targets = batch["tgt"]
        self.test_metric([item["tgt"] for item in preds["pred"]], targets)

        # if batch_idx == 0:
        #     self.print_prediction(batch)

        def split_prediction_string(s: str):
            buffer = []
            output = []
            for line in chain(s.split("\n"), ["==="]):
                if line.startswith("==="):
                    if len(buffer) > 0:
                        output.append("\n".join(buffer))
                        buffer.clear()
                elif len(line.strip()) > 0:
                    buffer.append(line)
            return output

        if self.hparams.export_detailed_prediction:
            device = batch["src_ids"].device
            str_io = StringIO()
            self.print_prediction(batch, handler=partial(print, file=str_io), batch_size=None)
            parses_on_given = split_prediction_string(str_io.getvalue())
            scores_on_given = preds["baseline"]
            parses_on_predicted = []
            for i, pred in enumerate(preds["pred"]):
                if (emp := batch.get("emphasize")) is not None:
                    pred["emp"] = emp[i, : len(pred["src"])]
                _batch = self.datamodule.collator([pred])
                _batch = self.datamodule.transfer_batch_to_device(_batch, device, 0)
                str_io = StringIO()
                self.print_prediction(_batch, handler=partial(print, file=str_io))
                parses_on_predicted.extend(split_prediction_string(str_io.getvalue()))
            scores_on_predicted = preds["score"]
            assert (
                len(parses_on_given) == len(scores_on_given) == len(parses_on_predicted) == len(scores_on_predicted)
            ), (len(parses_on_given), len(scores_on_given), len(parses_on_predicted), len(scores_on_predicted))
            detailed = list(zip(parses_on_given, scores_on_given, parses_on_predicted, scores_on_predicted))
        else:
            detailed = None

        return {
            "preds": [item["tgt"] for item in preds["pred"]],
            "detailed": detailed,
            "targets": targets,
            "id": batch["id"],
        }

    def test_epoch_end(self, outputs) -> None:
        acc = self.test_metric.compute()
        if not isinstance(acc, dict):
            acc = {"result": acc}
        d = {"test/" + k: v for k, v in acc.items()}
        d["step"] = self.global_step
        self.log_dict(d)
        self.print(acc)
        self.save_predictions(outputs)
        if outputs[0].get("detailed") is not None:
            self.save_detailed_predictions(outputs)


def make_subbatch(batch, size):
    output = {}
    for key, value in batch.items():
        if key == "transformer_inputs":
            output[key] = value
        elif key == "tgt_masks":
            output[key] = [item[:size] for item in value]
        elif key == "src_masks":
            output[key] = [item[:size] for item in value]
        else:
            output[key] = value[:size]
    return output
