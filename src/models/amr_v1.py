import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from prettytable import PrettyTable
from pytorch_lightning.profilers import PassThroughProfiler, SimpleProfiler
from torch_scatter import scatter_mean
from torch_struct.distributions import SentCFG
from torchmetrics import MaxMetric, MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.models.posterior_regularization.amr import AMRNeqTasks
from src.models.posterior_regularization.pr import compute_pr
from src.models.src_parser.base import SrcParserBase
from src.models.tgt_parser.base import TgtParserBase
from src.models.tree_encoder.base import TreeEncoderBase
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    get_actions,
    get_tree,
    report_ids_when_err,
)
from src.utils.metric import PerplexityMetric

from .general_seq2seq import GeneralSeq2SeqModule

log = logging.getLogger(__file__)


class AMRV1Module(GeneralSeq2SeqModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_neq_e_metric = MaxMetric()

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage, datamodule)
        self.amr_neq_pr_task = AMRNeqTasks[type(self.decoder)]
        self.profiler = self.trainer.profiler or PassThroughProfiler()

    @report_ids_when_err
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        with self.profiler.profile("compute_src_nll_and_sampling"):
            dist = self.parser(src_ids, src_lens)
            src_nll = -dist.partition
            src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist=dist)
            src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        with self.profiler.profile("src_encoding"):
            x = self.encode(batch)
            node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        with self.profiler.profile("compute_tgt_nll"):
            (
                params,
                pt_spans,
                pt_num_nodes,
                nt_spans,
                nt_num_nodes,
            ) = self.decoder.get_params(
                node_features,
                node_spans,
                batch["tgt_ids"],
                impossible_span_mask=batch["tgt_masks"],
                copy_position=copy_position,
            )
            tgt_nll = self.decoder(
                batch["tgt_ids"],
                batch["tgt_lens"],
                node_features,
                node_spans,
                params=params,
                impossible_span_mask=batch["tgt_masks"],
                copy_position=copy_position,
            )

        pt_neq_task = self.amr_neq_pr_task(
            pt_num_nodes,
            nt_num_nodes,
            self.decoder.pt_states,
            self.decoder.nt_states,
            tgt_nll.device,
        )
        if self.training:
            # TODO study whether we can avoid p.partition as it will be canceled by pt_neq_pr
            pt_neq_pr = compute_pr(
                params, batch["tgt_lens"], batch["tgt_pt_neq_constraint"], pt_neq_task
            ).mean()

            # # we have closed form of q
            # params = (params["term"], params["rule"], params["root"], params["copy_nt"])
            # pm = SentCFG(params, batch["tgt_lens"]).marginals[0]  # term marginals
            # pm = pm.view(*pm.shape[:2], -1, pt_num_nodes).sum(-2)
            pm = self.decoder.pcfg(params, batch["tgt_lens"], marginal=True).sum(3)
            pm = pm.diagonal(offset=1, dim1=1, dim2=2).transpose(1, 2)
            pt_eq_pr = []
            for bidx, groups in enumerate(batch["tgt_pt_eq_constraint"]):
                for group in groups:
                    d1 = pm[bidx].gather(
                        0, group.unsqueeze(-1).expand(-1, pm.shape[-1])
                    )
                    d1 = d1.clamp(1e-6)
                    if len(group) > 2:
                        d2 = torch.roll(d1, 0, dims=0)
                    else:
                        d1, d2 = torch.split(d1, 1, 0)
                    pt_eq_pr.append(-((d1.log() + d2.log()) / 2).logsumexp(-1).mean())
            if len(pt_eq_pr) > 0:
                pt_eq_pr = torch.stack(pt_eq_pr).mean()
                pr_terms = pt_neq_pr + pt_eq_pr
            else:
                pt_eq_pr = 0.0
                pr_terms = pt_neq_pr
            max_neq_e = -1
        else:
            pt_neq_pr, pt_eq_pr, pr_terms = 0.0, 0.0, 0.0
            pt_neq_e = pt_neq_task.calc_e(
                params, batch["tgt_lens"], batch["tgt_pt_neq_constraint"]
            )
            max_neq_e = pt_neq_e.max()

        with self.profiler.profile("src_reward"):
            with torch.no_grad():
                src_spans_argmax, src_logprob_argmax = self.parser.argmax(
                    src_ids, src_lens, dist=dist
                )
                src_spans_argmax = extract_parses(
                    src_spans_argmax[-1], src_lens, inc=1
                )[0]
                node_features_argmax, node_spans_argmax = self.tree_encoder(
                    x, src_lens, spans=src_spans_argmax
                )
                tgt_nll_argmax = self.decoder(
                    batch["tgt_ids"],
                    batch["tgt_lens"],
                    node_features_argmax,
                    node_spans_argmax,
                    copy_position=copy_position,
                    impossible_span_mask=batch["tgt_masks"],
                )
                neg_reward = (tgt_nll - tgt_nll_argmax).detach()

        return {
            "decoder": tgt_nll.mean() + pr_terms,
            "encoder": src_nll.mean() + (src_logprob * neg_reward).mean(),
            "tgt_nll": tgt_nll.mean(),
            "src_nll": src_nll.mean(),
            "pt_neq_pr": pt_neq_pr,
            "pt_eq_pr": pt_eq_pr,
            "reward": -neg_reward.mean(),
            "max_neq_e": max_neq_e,
        }

    def forward_visualize(self, batch, sample=False):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        parse = self.parser.sample if sample else self.parser.argmax
        src_spans = parse(src_ids, src_lens)[0][-1]
        src_spans, src_trees = extract_parses(src_spans, src_lens, inc=1)
        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        # show PRed trees
        (
            params,
            pt_spans,
            pt_num_nodes,
            nt_spans,
            nt_num_nodes,
        ) = self.decoder.get_params(
            node_features,
            node_spans,
            batch["tgt_ids"],
            # impossible_span_mask=batch["tgt_masks"],
            copy_position=copy_position,
        )
        pt_neq_task = self.amr_neq_pr_task(
            pt_num_nodes,
            nt_num_nodes,
            self.decoder.pt_states,
            self.decoder.nt_states,
            x.device,
        )
        print(
            pt_neq_task.calc_e(
                params, batch["tgt_lens"], batch["tgt_pt_neq_constraint"]
            )
        )
        params = compute_pr(
            params,
            batch["tgt_lens"],
            batch["tgt_pt_neq_constraint"],
            pt_neq_task,
            get_param=True,
        )
        params = (
            params,
            pt_spans,
            pt_num_nodes,
            nt_spans,
            nt_num_nodes,
        )
        print(
            pt_neq_task.calc_e(
                params[0], batch["tgt_lens"], batch["tgt_pt_neq_constraint"]
            )
        )

        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            params=params,
            copy_position=copy_position,
            # impossible_span_mask=batch["tgt_masks"],
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
                # only show token alignment
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
                    elif batch.get("copy_nt") is not None:
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

        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist=dist)[0]
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        y_preds = self.decoder.generate(
            node_features,
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

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/decoder", output["decoder"], prog_bar=True)
        self.log("train/pt_neq_pr", output["pt_neq_pr"], prog_bar=True)
        self.log("train/pt_eq_pr", output["pt_eq_pr"], prog_bar=True)
        self.log("train/encoder", output["encoder"], prog_bar=True)
        self.log("train/reward", output["reward"])

        if batch_idx == 0:
            self.eval()
            single_inst = {
                key: value[:2]
                for key, value in batch.items()
                if key not in {"tgt_masks"}
            }
            single_inst["tgt_masks"] = [item[:1] for item in batch["tgt_masks"]]
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg, neqs, eqs in zip(
                trees["src_tree"],
                trees["tgt_tree"],
                trees["alignment"],
                batch["tgt_pt_neq_constraint"],
                batch["tgt_pt_eq_constraint"],
            ):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                table = PrettyTable(["Src token", "Tgt token", "Is copy", "Neq", "Eq"])

                eq_flag = ["" for _ in alg]
                for gidx, group in enumerate(eqs):
                    gidx = str(gidx)
                    for item in group.tolist():
                        eq_flag[item] = gidx

                neq_flag = ["1" if item > 0 else "" for item in neqs.tolist()]
                # neq_flag += [''] * (len(alg) - len(neq_flag))  # NOTE implicitly trunc span

                for (s, t, c), n, e in zip(alg, neq_flag, eq_flag):
                    if t[0] == ":":
                        continue
                    table.add_row((s, t, c, n, e))
                self.print("\n", table, sep="")
            self.train()
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_metric.reset()
        self.val_neq_e_metric.reset()
        if self.hparams.track_param_norm:
            self.log(
                "stat/parser_norm",
                sum(param.norm() ** 2 for param in self.parser.parameters()).item()
                ** 0.5,
            )
            self.log(
                "stat/encoder_norm",
                sum(
                    param.norm() ** 2 for param in self.tree_encoder.parameters()
                ).item()
                ** 0.5,
            )
            self.log(
                "stat/decoder_norm",
                sum(param.norm() ** 2 for param in self.decoder.parameters()).item()
                ** 0.5,
            )

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])
        self.val_neq_e_metric(output["max_neq_e"])
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute()
        max_neq_e = self.val_neq_e_metric.compute()
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        # self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)
        self.log("val/max_neq_e", max_neq_e, on_epoch=True, prog_bar=True)
        self.print("val/ppl", str(ppl.item()))

    @torch.inference_mode(False)
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        # patch for inference_mode
        batch["src_ids"] = batch["src_ids"].clone()
        batch["tgt_ids"] = batch["tgt_ids"].clone()

        preds = self.forward_inference(batch)["pred"]
        targets = batch["tgt"]

        self.test_metric(preds, targets)

        if batch_idx == 0:
            single_inst = {
                key: value[:2]
                for key, value in batch.items()
                if key not in {"tgt_masks"}
            }
            single_inst["tgt_masks"] = [item[:2] for item in batch["tgt_masks"]]
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

        return {"preds": preds, "targets": targets, "id": batch["id"]}

    def test_epoch_end(self, outputs) -> None:
        acc = self.test_metric.compute()
        self.log_dict({"test/" + k: v for k, v in acc.items()})
        self.print(acc)
        self.save_predictions(outputs)
