import logging
import operator
import os
from copy import deepcopy
from functools import partial
from io import StringIO
from itertools import chain
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from pytorch_memlab import profile_every

from src.models.posterior_regularization.general import NeqPT, NoManyToOneNT
from src.models.posterior_regularization.pr import compute_pr
from src.models.rnng.rnng import GeneralRNNG
from src.utils.fn import annotate_snt_with_brackets, apply_to_nested_tensor, report_ids_when_err, spans2tree
from src.utils.metric import PerplexityMetric

from .general_seq2seq_end2end_struatt import GeneralSeq2SeqEnd2EndStruAttModule

log = logging.getLogger(__file__)


class PcfgPcfgrnngReinforceModel(GeneralSeq2SeqEnd2EndStruAttModule):
    def __init__(self, rnng, depth_reg, debug_1, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

    def setup_patch(self, stage: Optional[str] = None, datamodule=None):
        assert self.hparams.decoder_entropy_reg > 0.0
        self.rnng: GeneralRNNG = instantiate(
            self.hparams.rnng, input_dim=self.encoder.get_output_dim(), vocab=len(self.datamodule.tgt_vocab)
        )

        self.train_metric_rnng = PerplexityMetric()
        self.val_metric_rnng = PerplexityMetric()
        self.test_metric_rnng = deepcopy(self.test_metric)

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

        with self.profiler.profile("compute_src_nll_and_marginal"):
            dist = self.parser(src_ids, src_lens, extra_scores=extra_scores)
            src_loss = src_nll = dist.nll
            term_m, span_m = dist.marginal_with_grad

        with self.profiler.profile("src_encoding"):
            seq_h, span_h = self.encode(batch)
            node_features, node_spans, weight = self.build_node_feature(span_h, src_lens, term_m, span_m)

        with self.profiler.profile("compute_tgt_nll"):
            tgt_pred = self.decoder(node_features, node_spans, weight)
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            tgt_loss = tgt_nll = tgt_pred.dist.nll

        with self.profiler.profile("sample_tgt_pcfg"):
            tgt_sampled_tree = tgt_pred.dist.sample_one(need_event=True, need_span=True)
            self.rnng.setup_nt(tgt_pred.nt_features, tgt_pred.nt_num_nodes)
            action_ids, stack_size = self.rnng.make_batch(tgt_sampled_tree["span"], tgt_lens, tgt_ids.device)
            rnng_loss, a_loss, w_loss, _ = self.rnng(tgt_ids, action_ids, max(stack_size))
            logging_vals["rnng_action_loss"] = a_loss
            logging_vals["rnng_word_loss"] = w_loss

        with self.profiler.profile("qcfg_reward"), torch.no_grad():
            tgt_argmax_tree = tgt_pred.dist.decoded
            action_ids, stack_size_argmax = self.rnng.make_batch(tgt_argmax_tree, tgt_lens, tgt_ids.device)
            rnng_loss_baseline, _, _, _ = self.rnng(tgt_ids, action_ids, max(stack_size_argmax))
            neg_reward = rnng_loss - rnng_loss_baseline
            logging_vals["rnng_reward"] = -neg_reward

        sample_logprob = tgt_pred.dist.score(tgt_sampled_tree["event"])
        logging_vals["sample_logprob"] = sample_logprob

        if self.hparams.depth_reg > 0:
            depth = self.compute_tree_height(tgt_sampled_tree["span"])
            # depth_argmax = self.compute_tree_height(tgt_argmax_tree)
            depth_baseline = torch.log2(torch.tensor(tgt_lens, dtype=torch.float32))
            _depth_neg_reward = (depth - depth_baseline).to(tgt_ids.device)
            logging_vals["depth_reward"] = -_depth_neg_reward
            neg_reward += self.hparams.depth_reg * _depth_neg_reward

        if self.hparams.debug_1:
            objective = 0.0
        else:
            objective = sample_logprob * neg_reward
        # objective = (sample_logprob - tgt_nll) * neg_reward

        soft_constraint_pr_loss = 0
        src_entropy_reg = 0
        tgt_entropy_reg = 0
        pr_neq_pt_reg = 0
        pr_neq_nt_reg = 0

        if self.training and self.current_epoch >= self.hparams.warmup_qcfg:

            if self.hparams.soft_constraint_loss_rl:
                tgt_loss = self.decoder.get_rl_loss(tgt_pred, self.hparams.decoder_entropy_reg)
            if self.hparams.soft_constraint_loss_raml:
                tgt_loss = self.decoder.get_raml_loss(tgt_pred)

            if (e := self.hparams.soft_constraint_loss_pr) > 0:
                soft_constraint_pr_loss = self.decoder.get_soft_constraint_loss(tgt_pred)
                logging_vals["soft_constraint_pr"] = soft_constraint_pr_loss
                soft_constraint_pr_loss = e * soft_constraint_pr_loss

            if (e := self.hparams.pr_pt_neq_reg) > 0:
                l = compute_pr(tgt_pred, None, NeqPT())
                logging_vals["pr_neq_pt"] = l
                pr_neq_pt_reg = l * e
            if (e := self.hparams.pr_nt_neq_reg) > 0:
                l = compute_pr(tgt_pred, None, NoManyToOneNT())
                logging_vals["pr_neq_nt"] = l
                pr_neq_nt_reg = l * e

            if (e := self.hparams.parser_entropy_reg) > 0:
                entropy = self.parser.entropy(src_ids, src_lens, dist=dist)
                src_entropy_reg = -e * entropy
                logging_vals["src_entropy"] = entropy
            if (e := self.hparams.decoder_entropy_reg) > 0:
                entropy = tgt_pred.dist.entropy
                tgt_entropy_reg = -e * entropy
                logging_vals["tgt_entropy"] = entropy

        return {
            "decoder": tgt_loss
            + objective
            + soft_constraint_pr_loss
            + tgt_entropy_reg
            + pr_neq_pt_reg
            + pr_neq_nt_reg,
            "encoder": src_loss + src_entropy_reg,
            "rnng": rnng_loss,
            "tgt_nll": tgt_nll,
            "src_nll": src_nll,
            "rnng_nll": rnng_loss,
            "runtime": {"seq_encoded": seq_h, "span_encoded": span_h},
            "src_runtime": {
                "dist": dist,
            },
            "tgt_runtime": {"pred": tgt_pred},
            "log": logging_vals,
        }

    def compute_tree_height(self, spans):
        depths = []
        for b, nt_spans_inst in enumerate(spans):
            parents = spans2tree(nt_spans_inst)
            nt_depths = []
            for j in range(len(nt_spans_inst)):
                depth = 0
                k = parents[j]
                while k != -1:
                    k = parents[k]
                    depth += 1
                nt_depths.append(depth)
            depths.append(max(nt_depths))
        return torch.tensor(depths, dtype=torch.float32)

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
        }

        dist = self.parser(src_ids, src_lens, extra_scores=extra_scores)

        src_spans = self.parser.argmax(src_ids, src_lens, extra_scores=extra_scores, dist=dist)
        src_annotated = []
        for snt, src_span_inst in zip(batch["src"], src_spans):
            tree = annotate_snt_with_brackets(snt, src_span_inst)
            src_annotated.append(tree)

        # TODO use argmax?
        term_m = dist.marginal["term"]
        span_m = dist.marginal["trace"]

        seq_h, span_h = self.encode(batch)
        node_features, node_spans, weight = self.build_node_feature(span_h, src_lens, term_m, span_m)

        tgt_pred = self.decoder(node_features, node_spans, weight)
        tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(tgt_pred)
        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)

        self.rnng.setup_nt(tgt_pred.nt_features, tgt_pred.nt_num_nodes)
        rnng_pred = self.rnng.decode(batch["tgt"], batch["tgt_ids"])

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
            "rnng_tree": rnng_pred,
            "alignment": alignments,
        }

    @report_ids_when_err
    def forward_generate(self, batch, get_baseline=False):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        tgt_ids, tgt_lens = batch["tgt_ids"], batch["tgt_lens"]

        dist = self.parser(src_ids, src_lens)
        term_m = dist.marginal["term"]
        span_m = dist.marginal["trace"]

        seq_h, span_h = self.encode(batch)
        node_features, node_spans, weight = self.build_node_feature(span_h, src_lens, term_m, span_m)

        tgt_pred = self.decoder(node_features, node_spans, weight)
        tgt_pred = self.decoder.prepare_sampler(tgt_pred, batch["src"], src_ids)
        y_preds = self.decoder.generate(tgt_pred)

        self.rnng.setup_nt(tgt_pred.nt_features, tgt_pred.nt_num_nodes)
        self.rnng.generate(batch_size=len(src_ids), max_length=30, beam_size=5, device=src_ids.device)

        if get_baseline:

            t = self.decoder.observe_x(tgt_pred, tgt_ids, tgt_lens)
            self.rnng.setup_nt(t.nt_features, t.nt_num_nodes)
            action_ids, stack_size_argmax = self.rnng.make_batch(t.dist.decoded, tgt_lens, tgt_ids.device)
            ll, _, _, _ = self.rnng(tgt_ids, action_ids, max(stack_size_argmax))
            rnng_baseline_ppl = np.exp(ll.detach().cpu().numpy() / np.array(tgt_lens)).tolist()

            pb = [item[0] for item in y_preds]
            pb = self.datamodule.collator(pb)
            pb = self.datamodule.transfer_batch_to_device(pb, src_ids.device, 0)
            pt = self.decoder.observe_x(tgt_pred, pb["tgt_ids"], pb["tgt_lens"])
            self.rnng.setup_nt(pt.nt_features, pt.nt_num_nodes)
            action_ids, stack_size_argmax = self.rnng.make_batch(pt.dist.decoded, pb["tgt_lens"], src_ids.device)
            ll, _, _, _ = self.rnng(pb["tgt_ids"], action_ids, max(stack_size_argmax))
            rnng_ppl = np.exp(ll.detach().cpu().numpy() / np.array(pb["tgt_lens"])).tolist()

            observed = {
                "x": batch["tgt_ids"],
                "lengths": batch["tgt_lens"],
                "pt_copy": batch.get("copy_token"),
                "nt_copy": batch.get("copy_phrase"),
            }
            tgt_pred = self.decoder.observe_x(tgt_pred, **observed)
            baseline = np.exp(tgt_pred.dist.nll.detach().cpu().numpy() / np.array(tgt_lens)).tolist()
        else:
            rnng_baseline_ppl = None
            rnng_ppl = None
            baseline = None

        return {
            "pred": [item[0] for item in y_preds],
            "score": [item[1] for item in y_preds],
            "rnng_score": rnng_ppl,
            "baseline": baseline,
            "rnng_baseline": rnng_baseline_ppl,
        }

    def print_prediction(self, batch, handler=None, batch_size=1):
        training_state = self.training
        self.train(False)
        if handler is None:
            handler = self.print
        if isinstance(batch_size, int):
            batch = make_subbatch(batch, batch_size)
        trees = self.forward_visualize(batch)
        for src, tgt, tgt_rnng, alg in zip(
            trees["src_tree"], trees["tgt_tree"], trees["rnng_tree"], trees["alignment"]
        ):
            handler("Src:      ", src)
            handler("Tgt:      ", tgt)
            handler("RNNG Tgt: ", tgt_rnng)
            handler("Alignment:\n" + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg)))
            handler("=" * 79)
        self.train(training_state)

    def save_detailed_predictions(self, outputs, path=None):
        if path is None:
            path = "detailed_predict_on_test.txt"
            if os.path.exists(path):
                for i in range(2, 1000):
                    path = f"detailed_predict_on_test_{i}.txt"
                    if not os.path.exists(path):
                        break
        log.info(f"Writing to {os.path.abspath(path)}")
        if dist.is_initialized() and (ws := dist.get_world_size()):
            if len(self.datamodule.data_test) % ws != 0:
                log.warning(
                    "Do NOT report the above metrics because you are using"
                    "DDP and the size of the testset is odd, which means"
                    "there is one sample counted twice due to the"
                    "DistributedSampler. Run evaluation on the"
                    "predict_on_test.txt file"
                )
            merged = [None] * ws
            dist.all_gather_object(merged, outputs)
            if self.global_rank == 0:
                outputs = sum(merged, [])

        if self.global_rank == 0:
            preds = []
            for inst in outputs:
                preds_batch = inst["detailed"]
                id_batch = inst["id"].tolist()
                preds.extend(zip(id_batch, preds_batch))
            preds.sort(key=lambda x: x[0])

            # remove duplicate
            to_remove = []
            for i in range(1, len(preds)):
                if preds[i - 1][0] == preds[i][0]:
                    to_remove.append(i)
            for i in reversed(to_remove):
                del preds[i]

            # check missing
            if not (preds[0][0] == 0 and preds[-1][0] == len(preds) - 1):
                log.warning(f"There are some missing examples. Last id={preds[-1][0]}. Len={len(preds)}")

            with open(path, "w") as f:
                for id_, inst in preds:
                    # f.write(f"{id_}:\t")
                    f.write(">>> [Parse on gold target sequence] " + ">" * 33)
                    f.write("\n")
                    f.write(f"Score:\t{inst[1]}")
                    f.write("\n")
                    f.write(f"Rnng Score:\t{inst[4]}")
                    f.write("\n")
                    f.write(inst[0])
                    f.write("\n")
                    f.write("-" * 70)
                    f.write("\n")
                    f.write(f"Score:\t{inst[3]}")
                    f.write("\n")
                    f.write(f"Rnng Score:\t{inst[5]}")
                    f.write("\n")
                    f.write(inst[2])
                    f.write("\n")
                    f.write("<<< [Parse on predicted sequence] " + "<" * 35)
                    f.write("\n\n")

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def training_step(self, batch: Any, batch_idx: int, *, forward_prediction=None):
        output = forward_prediction if forward_prediction is not None else self(batch)
        loss_decoder = output["decoder"].mean()
        loss_encoder = output["encoder"].mean()
        loss_rnng = output["rnng"].mean()
        loss = loss_encoder + loss_decoder + loss_rnng
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        ppl_rnng = self.train_metric_rnng(output["rnng_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/ppl_rnng", ppl_rnng, prog_bar=True)
        self.log("train/src", loss_encoder, prog_bar=True)
        self.log("train/tgt", loss_decoder, prog_bar=True)
        self.log("train/rnng", loss_rnng, prog_bar=True)

        if "log" in output:
            self.log_dict({"train/" + k: v.mean() for k, v in output["log"].items()})

        if batch_idx % 100 == 0:
            self.print_prediction(batch, batch_size=2)

        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_metric.reset()
        self.val_metric_rnng.reset()
        self.test_metric.reset()
        self.test_metric_rnng.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss_decoder = output["decoder"].mean()
        loss_encoder = output["encoder"].mean()
        loss_rnng = output["rnng"].mean()
        loss = loss_encoder + loss_decoder + loss_rnng
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])
        self.val_metric_rnng(output["rnng_nll"], batch["tgt_lens"])

        if (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0 and self.current_epoch > self.warmup:
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

        if (self.current_epoch + 1) % self.hparams.real_val_every_n_epochs == 0 and self.current_epoch > self.warmup:
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
        self.test_metric_rnng.reset()
        return super().on_test_epoch_start()

    @torch.inference_mode(False)
    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        # patch for inference_mode
        batch = apply_to_nested_tensor(batch, func=lambda x: x.clone())

        preds = self.forward_generate(batch, get_baseline=self.hparams.export_detailed_prediction)
        targets = batch["tgt"]
        # targets_rnng = batch['tgt_rnng']
        self.test_metric([item["tgt"] for item in preds["pred"]], targets)
        # self.test_metric_rnng(targets_rnng, targets)

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
            rnng_scores_on_given = preds["rnng_baseline"]
            parses_on_predicted = []
            for pred in preds["pred"]:
                _batch = self.datamodule.collator([pred])
                _batch = self.datamodule.transfer_batch_to_device(_batch, device, 0)
                str_io = StringIO()
                self.print_prediction(_batch, handler=partial(print, file=str_io))
                parses_on_predicted.extend(split_prediction_string(str_io.getvalue()))
            scores_on_predicted = preds["score"]
            rnng_scores_on_predicted = preds["rnng_score"]
            assert (
                len(parses_on_given)
                == len(scores_on_given)
                == len(parses_on_predicted)
                == len(scores_on_predicted)
                == len(rnng_scores_on_given)
                == len(rnng_scores_on_predicted)
            ), (
                len(parses_on_given),
                len(scores_on_given),
                len(parses_on_predicted),
                len(scores_on_predicted),
                len(rnng_scores_on_given),
                len(rnng_scores_on_predicted),
            )
            detailed = list(
                zip(
                    parses_on_given,
                    scores_on_given,
                    parses_on_predicted,
                    scores_on_predicted,
                    rnng_scores_on_given,
                    rnng_scores_on_predicted,
                )
            )
        else:
            detailed = None

        return {
            "preds": [item["tgt"] for item in preds["pred"]],
            # "preds_rnng": targets_rnng,
            "detailed": detailed,
            "targets": targets,
            "id": batch["id"],
        }

    def test_epoch_end(self, outputs) -> None:
        metric = self.test_metric.compute()
        # metric_rnng = self.test_metric_rnng.compute()
        if not isinstance(metric, dict):
            metric = {"result": metric}
        # if not isinstance(metric_rnng, dict):
        # metric_rnng = {"result": metric_rnng}
        d = {
            "test/" + k: v for k, v in metric.items()
        }  # | {"test/" + k + "_rnng": v for k, v in metric_rnng.items()}
        d["step"] = self.global_step
        self.log_dict(d)
        self.print(metric)
        # self.print(metric_rnng)
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
