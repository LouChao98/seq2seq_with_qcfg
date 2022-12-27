import logging
from typing import Any, List, Optional

import torch
import wandb
from hydra.utils import instantiate
from pytorch_memlab import profile_every
from torch.autograd import grad

from src.models.posterior_regularization.general import NeqNT, NeqPT
from src.models.posterior_regularization.pr import compute_pr
from src.models.struct.semiring import LogSemiring
from src.utils.fn import apply_to_nested_tensor, report_ids_when_err

from .general_seq2seq import GeneralSeq2SeqModule

log = logging.getLogger(__file__)


class GeneralSeq2SeqAutoencoderModule(GeneralSeq2SeqModule):
    def __init__(self, reconstructor=None, load_reconstructor_from_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage, datamodule)
        if self.hparams.load_reconstructor_from_checkpoint is not None:
            log.info(f"Loading reconstructor from {self.hparams.load_reconstructor_from_checkpoint}")
            with open(self.hparams.load_reconstructor_from_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")["state_dict"]
            loadedkeys = self.reconstructor.load_state_dict(state_dict, strict=False)
            if len(loadedkeys.unexpected_keys) > 0:
                log.warning(f"Unexpected keys: {loadedkeys.unexpected_keys}")

        if wandb.run is not None:
            wandb.run.tags = wandb.run.tags + (type(self.reconstructor).__name__,)

    def setup_patch(self, stage: Optional[str] = None, datamodule=None):
        self.reconstructor = instantiate(
            self.hparams.reconstructor, datamodule=datamodule, src_node_dim=self.tree_encoder.get_output_dim()
        )

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
            # span labels are discarded (set to -1)
            node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        if self.current_epoch < self.hparams.warmup_pcfg:
            node_features = apply_to_nested_tensor(node_features, lambda x: torch.zeros_like(x))

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred_orig = self.decoder(node_features, node_spans)
            tgt_pred = self.decoder.observe_x(tgt_pred_orig, **observed, inplace=False)
            tgt_loss = tgt_nll = tgt_pred.dist.nll

            if self.decoder.use_copy:
                use_copy, self.decoder.use_copy = self.decoder.use_copy, False
                tgt_pred_uc = self.decoder.observe_x(tgt_pred_orig, tgt_ids, tgt_lens, inplace=False)
                self.decoder.use_copy = use_copy
            else:
                tgt_pred_uc = tgt_pred

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
        ae_loss = 0

        if self.training and self.current_epoch < self.hparams.warmup_qcfg:
            ae_loss, to_log = self.reconstructor.forward_lang_only(batch, tgt_pred_uc)
            logging_vals.update(to_log)

        if self.training and self.current_epoch >= self.hparams.warmup_qcfg:
            ae_loss, to_log = self.reconstructor(batch, tgt_pred_uc)
            logging_vals["ae"] = ae_loss
            logging_vals.update(to_log)

            if "rec_neg_reward" in logging_vals:
                objective += src_logprob * logging_vals["rec_neg_reward"]

            # print(to_log['vae_beta'])
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

            if (e := self.hparams.pr_pt_neq_reg) > 0:
                l = compute_pr(tgt_pred, None, NeqPT())
                logging_vals["pr_neq_pt"] = l
                pr_neq_pt_reg = l * e
            if (e := self.hparams.pr_nt_neq_reg) > 0:
                l = compute_pr(tgt_pred, None, NeqNT())
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

        return {
            "decoder": tgt_loss
            + length_calibrate_term
            + soft_constraint_pr_loss
            + tgt_entropy_reg
            + noisy_span_loss
            + pr_neq_pt_reg
            + pr_neq_nt_reg
            + pt_prior_loss
            + ae_loss,
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
