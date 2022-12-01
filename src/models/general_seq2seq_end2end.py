import logging
import operator
from typing import Optional

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

from src.models.posterior_regularization.general import NeqNT, NeqPT
from src.models.posterior_regularization.pr import compute_pr
from src.models.src_parser.base import SrcParserBase
from src.models.struct.semiring import LogSemiring
from src.models.tgt_parser.base import TgtParserBase
from src.utils.fn import annotate_snt_with_brackets, report_ids_when_err
from src.utils.metric import PerplexityMetric

from .general_seq2seq import GeneralSeq2SeqModule

log = logging.getLogger(__file__)


class GeneralSeq2SeqEnd2EndModule(GeneralSeq2SeqModule):
    def __init__(self, method="gumbel", no_src_nll=False, **kwargs):
        assert method in ("gumbel", "argmax")
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        assert self.hparams.tree_encoder is None

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
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
        self.decoder: TgtParserBase = instantiate(
            self.hparams.decoder,
            vocab=len(self.datamodule.tgt_vocab),
            datamodule=self.datamodule,
            src_dim=self.encoder.get_output_dim(),
        )

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
                if name.startswith("pretrained."):
                    continue
                if param.dim() > 1:
                    init_func(param)
                elif "norm" not in name.lower():
                    nn.init.zeros_(param)

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
        hidden_size = x.shape[-1]
        x = torch.cat(
            [src_ids.new_zeros(len(src_ids), 1, hidden_size), x, src_ids.new_zeros(len(src_ids), 1, hidden_size)],
            dim=1,
        )
        x[torch.arange(len(src_ids)), torch.tensor(src_lens) + 1] = 0.0
        x = self.encoder(x, [item + 2 for item in src_lens])
        seq_h = x[:, 1:-1]
        x = torch.cat(
            [
                x[:, :-1, : hidden_size // 2],
                x[:, 1:, hidden_size // 2 :],
            ],
            -1,
        )
        span_h = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
        return seq_h, span_h

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
            "observed_mask": batch.get("tgt_masks"),
        }
        logging_vals = {}

        with self.profiler.profile("compute_src_nll_and_sampling"):
            dist = self.parser(src_ids, src_lens, extra_scores=extra_scores)
            if self.hparams.method == "gumbel":
                span_indicator = dist.gumbel_sample_one(1.0)
            else:
                span_indicator = dist.argmax_st()
            src_loss = src_nll = dist.nll

        with self.profiler.profile("src_encoding"):
            seq_h, span_h = self.encode(batch)

        if self.current_epoch < self.hparams.warmup_pcfg:
            seq_h = torch.zeros_like(seq_h)
            span_h = torch.zeros_like(span_h)

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred = self.decoder(seq_h, span_h, span_indicator, src_lens)
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            tgt_loss = tgt_nll = tgt_pred.dist.nll

        soft_constraint_pr_loss = 0
        noisy_span_loss = 0
        pt_prior_loss = 0
        src_entropy_reg = 0
        tgt_entropy_reg = 0
        pr_neq_pt_reg = 0
        pr_neq_nt_reg = 0
        length_calibrate_term = 0
        if self.training and self.current_epoch >= self.hparams.warmup_qcfg:
            if self.hparams.soft_constraint_loss_rl:
                tgt_loss = self.decoder.get_rl_loss(tgt_pred)
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
            + pt_prior_loss,
            "encoder": (torch.zeros_like(src_loss) if self.hparams.no_src_nll else src_loss) + src_entropy_reg,
            "tgt_nll": tgt_nll,
            "src_nll": src_nll,
            "runtime": {"seq_encoded": seq_h, "span_encoded": span_h},
            "src_runtime": {
                "dist": dist,
            },
            "tgt_runtime": {"pred": tgt_pred},
            "log": logging_vals,
        }

    @report_ids_when_err
    def forward_visualize(self, batch):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        observed = {
            "x": batch["tgt_ids"],
            "lengths": batch["tgt_lens"],
            "pt_copy": batch.get("copy_token"),
            "nt_copy": batch.get("copy_phrase"),
            # "prior_alignment": batch.get("prior_alignment"),
            # "observed_mask": batch.get('tgt_masks')
        }

        src_spans = self.parser.argmax(src_ids, src_lens)
        src_annotated = []
        for snt, src_span_inst in zip(batch["src"], src_spans):
            tree = annotate_snt_with_brackets(snt, src_span_inst)
            src_annotated.append(tree)

        seq_h, span_h = self.encode(batch)
        span_indicator = torch.zeros(list(span_h.shape[:-1]) + [span_h.shape[1] - 2])
        for bidx, spans in enumerate(src_spans):
            spans.sort(key=lambda x: x[1] - x[0])
            span_idx = 0
            for i, (l, r, *_) in enumerate(spans):
                if l + 1 < r:
                    span_indicator[bidx, l, r, span_idx] = 1.0
                    span_idx += 1
        span_indicator = span_indicator.to(span_h.device)

        tgt_pred = self.decoder(seq_h, span_h, span_indicator, src_lens)
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

        seq_h, span_h = self.encode(batch)
        span_indicator = torch.zeros(list(span_h.shape[:-1]) + [span_h.shape[1] - 2])
        for bidx, spans in enumerate(src_spans):
            spans.sort(key=lambda x: x[1] - x[0])
            span_idx = 0
            for i, (l, r, *_) in enumerate(spans):
                if l + 1 < r:
                    span_indicator[bidx, l, r, span_idx] = 1.0
                    span_idx += 1
        span_indicator = span_indicator.to(span_h.device)

        tgt_pred = self.decoder(seq_h, span_h, span_indicator, src_lens)
        tgt_pred = self.decoder.prepare_sampler(tgt_pred, batch["src"], src_ids)
        y_preds = self.decoder.generate(tgt_pred)

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
