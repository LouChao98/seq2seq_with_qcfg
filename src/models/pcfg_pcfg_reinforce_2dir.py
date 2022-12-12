import logging
import operator
from copy import deepcopy
from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from torchmetrics import Metric, MinMetric

from src.models.base import ModelBase
from src.models.components.dynamic_hp import DynamicHyperParameter
from src.models.tgt_parser.base import NO_COPY_SPAN, TgtParserBase, TgtParserPrediction
from src.utils.fn import annotate_snt_with_brackets
from src.utils.metric import PerplexityMetric

from .general_seq2seq import GeneralSeq2SeqModule
from .posterior_regularization.agree import PTAgree

log = logging.getLogger(__file__)


def smoothed_hinge_loss(d, sigma):
    return torch.where(d.abs() < sigma, d**2 / (2 * sigma), d.abs()).flatten(1).sum(1)
    # return torch.where(d.abs() < sigma, d ** 2 / (2 * sigma), d.abs() - sigma).flatten(1).sum(1)


class _Unidirectional(GeneralSeq2SeqModule):
    # assume pt span range=[1,1], nt span range=[2, +\infty]
    def __init__(self, embedding, encoder, tree_encoder, parser, decoder, metric, hparams) -> None:
        ModelBase.__init__(self)
        self.hparams.update(hparams)
        self.warmup = 0

        self.embedding = embedding
        self.encoder = encoder
        self.tree_encoder = tree_encoder
        self.parser: TgtParserBase = parser
        self.decoder: TgtParserBase = decoder
        self.datamodule = self.decoder.datamodule

        self.dummy_node_emb = nn.Parameter(torch.randn(3, self.tree_encoder.get_output_dim()))

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric: Metric = deepcopy(metric)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        raise NotImplementedError("This class is managed by the outer class")

    def encode(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)
        x = self.encoder(x, src_lens)
        return x

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        tgt_ids, tgt_lens = batch["tgt_ids"], batch["tgt_lens"]
        batch_size = len(src_ids)
        logging_vals = {}

        x = self.encode(batch)
        src_shared_node_features = [self.dummy_node_emb for _ in range(batch_size)]
        src_shared_spans = [
            [(0, 1, NO_COPY_SPAN), (1, 2, NO_COPY_SPAN), (0, 2, NO_COPY_SPAN)] for _ in range(batch_size)
        ]

        with self.parser.disable_constraint():
            src_pcfg_pred = self.parser(src_shared_node_features, src_shared_spans)
            src_pcfg_pred: TgtParserPrediction = self.parser.observe_x(src_pcfg_pred, src_ids, src_lens)
        src_loss = src_nll = src_pcfg_pred.dist.nll
        src_sample = src_pcfg_pred.dist.sample_one(True, True)
        src_spans = src_sample["span"]
        src_logprob = src_pcfg_pred.dist.score(src_sample["event"])
        src_spans_argmax = src_pcfg_pred.dist.decoded

        src_node_features, src_node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_qcfg_pred = self.decoder(src_node_features, src_node_spans)
        tgt_qcfg_pred = self.decoder.observe_x(tgt_qcfg_pred, tgt_ids, tgt_lens)
        tgt_loss = tgt_nll = tgt_qcfg_pred.dist.nll

        with torch.no_grad():
            node_features_argmax, node_spans_argmax = self.tree_encoder(x, src_lens, spans=src_spans_argmax)
            tgt_argmax_pred = self.decoder(node_features_argmax, node_spans_argmax)
            tgt_argmax_pred = self.decoder.observe_x(tgt_argmax_pred, tgt_ids, tgt_lens)
            tgt_nll_argmax = tgt_argmax_pred.dist.nll
            fw_neg_reward = tgt_qcfg_pred.dist.nll - tgt_nll_argmax
            logging_vals["reward"] = -fw_neg_reward
        objective = src_logprob * fw_neg_reward

        return {
            "decoder": tgt_loss,
            "encoder": src_loss + objective,
            "tgt_runtime": {"pred": tgt_qcfg_pred},
            "src_nll": src_nll,
            "tgt_nll": tgt_nll,
            "log": logging_vals,
        }

    def forward_visualize(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        tgt_ids, tgt_lens = batch["tgt_ids"], batch["tgt_lens"]
        batch_size = len(src_ids)

        x = self.encode(batch)
        src_shared_node_features = [self.dummy_node_emb for _ in range(batch_size)]
        src_shared_spans = [
            [(0, 1, NO_COPY_SPAN), (1, 2, NO_COPY_SPAN), (0, 2, NO_COPY_SPAN)] for _ in range(batch_size)
        ]

        with self.parser.disable_constraint():
            src_pcfg_pred = self.parser(src_shared_node_features, src_shared_spans)
            src_pcfg_pred: TgtParserPrediction = self.parser.observe_x(src_pcfg_pred, src_ids, src_lens)
        src_spans = src_pcfg_pred.dist.decoded

        src_node_features, src_node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_qcfg_pred = self.decoder(src_node_features, src_node_spans)
        tgt_qcfg_pred = self.decoder.observe_x(tgt_qcfg_pred, tgt_ids, tgt_lens)
        tgt_spans, aligned_spans, pt_spans, nt_spans = self.decoder.parse(tgt_qcfg_pred)

        src_annotated = []
        for snt, src_span_inst in zip(batch["src"], src_spans):
            tree = annotate_snt_with_brackets(snt, src_span_inst)
            src_annotated.append(tree)

        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)

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
            # for tgt_span, src_span in zip(tgt_spans_inst, aligned_spans_inst):
            for i in idx:
                tgt_span = tgt_spans_inst[i]
                src_span = aligned_spans_inst[i]
                if src_span is None:
                    continue

                alignments_inst.append(
                    (
                        " ".join(src_snt[src_span[0] : src_span[1]]) + f" ({src_span[0]}, {src_span[1]})",
                        " ".join(tgt_snt[tgt_span[0] : tgt_span[1]]) + f" ({tgt_span[0]}, {tgt_span[1]})",
                        "",
                    )
                )
            alignments.append(alignments_inst[::-1])

        return {
            "src_tree": src_annotated,
            "tgt_tree": tgt_annotated,
            "alignment": alignments,
        }

    def forward_generate(self, batch, get_baseline=False):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        tgt_ids, tgt_lens = batch["tgt_ids"], batch["tgt_lens"]
        batch_size = len(src_ids)

        x_src = self.encode(batch)
        src_shared_node_features = [self.dummy_node_emb for _ in range(batch_size)]
        src_shared_spans = [
            [(0, 1, NO_COPY_SPAN), (1, 2, NO_COPY_SPAN), (0, 2, NO_COPY_SPAN)] for _ in range(batch_size)
        ]

        with self.parser.disable_constraint():
            src_pcfg_pred = self.parser(src_shared_node_features, src_shared_spans)
            src_pcfg_pred: TgtParserPrediction = self.parser.observe_x(src_pcfg_pred, src_ids, src_lens)
        src_spans = src_pcfg_pred.dist.decoded

        src_node_features, src_node_spans = self.tree_encoder(x_src, src_lens, spans=src_spans)

        tgt_qcfg_pred = self.decoder(src_node_features, src_node_spans)
        tgt_qcfg_pred = self.decoder.prepare_sampler(tgt_qcfg_pred, batch["src"], src_ids)
        tgt_preds = self.decoder.generate(tgt_qcfg_pred)

        if get_baseline:
            # TODO this always be ppl. but scores_on_predicted can be others
            tgt_pred = self.decoder.observe_x(tgt_qcfg_pred, tgt_ids, tgt_lens)
            baseline = np.exp(tgt_pred.dist.nll.detach().cpu().numpy() / np.array(tgt_lens)).tolist()
        else:
            baseline = None

        return {
            "pred": [item[0] for item in tgt_preds],
            "score": [item[1] for item in tgt_preds],
            "baseline": baseline,
        }


def _save_prediction(data, path=None, func=None, prefix=None):
    if path is None:
        return func(data, prefix + "predict_on_test.txt")
    else:
        # TODO allow folder
        return func(data, prefix + path)


def _save_detailed_prediction(data, path=None, func=None, prefix=None):
    if path is None:
        return func(data, prefix + "detailed_predict_on_test.txt")
    else:
        # TODO allow folder
        return func(data, prefix + path)


class PPRTwoDirModule(ModelBase):
    def __init__(
        self,
        embedding=None,
        transformer_pretrained_model=None,
        fix_pretrained=True,
        encoder=None,
        tree_encoder=None,
        decoder=None,
        optimizer=None,
        scheduler=None,
        test_metric=None,
        load_from_checkpoint=None,
        param_initializer="xavier_uniform",
        real_val_every_n_epochs=5,
        export_detailed_prediction=True,
        # extension
        train_pt_agreement=False,
        train_pt_agreement_strength=1.0,
        warmup_pcfg=0,
        warmup_qcfg=0,
        length_calibrate=False,
        pr_pt_neq_reg=0.0,
        pr_nt_neq_reg=0.0,
        noisy_spans_reg=0.0,
        noisy_spans_num=0.0,
        parser_entropy_reg=0.0,
        decoder_entropy_reg=0.0,
        soft_constraint_loss_pr=0.0,
        soft_constraint_loss_rl=False,
        soft_constraint_loss_raml=False,
    ):

        # real_val_every_n_epochs = 1
        assert warmup_pcfg <= warmup_qcfg
        self.warmup = warmup_qcfg
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.pt_constraint_strength = DynamicHyperParameter(train_pt_agreement_strength)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        embedding_src = instantiate(self.hparams.embedding, num_embeddings=len(self.datamodule.src_vocab))
        embedding_tgt = instantiate(self.hparams.embedding, num_embeddings=len(self.datamodule.tgt_vocab))
        encoder_src = instantiate(self.hparams.encoder, input_dim=embedding_src.weight.shape[1])
        encoder_tgt = instantiate(self.hparams.encoder, input_dim=embedding_tgt.weight.shape[1])
        tree_encoder_src = instantiate(self.hparams.tree_encoder, dim=encoder_src.get_output_dim())
        tree_encoder_tgt = instantiate(self.hparams.tree_encoder, dim=encoder_tgt.get_output_dim())
        parser_src = instantiate(
            self.hparams.decoder,
            vocab=len(self.datamodule.src_vocab),
            datamodule=self.datamodule,
            src_dim=tree_encoder_src.get_output_dim(),
        )
        parser_tgt = instantiate(
            self.hparams.decoder,
            vocab=len(self.datamodule.tgt_vocab),
            datamodule=self.datamodule,
            src_dim=tree_encoder_tgt.get_output_dim(),
        )
        test_metric: Metric = instantiate(self.hparams.test_metric)

        self.fw_model = _Unidirectional(
            embedding_src, encoder_src, tree_encoder_src, parser_src, parser_tgt, test_metric, self.hparams
        )
        self.bw_model = _Unidirectional(
            embedding_tgt, encoder_tgt, tree_encoder_tgt, parser_tgt, parser_src, test_metric, self.hparams
        )

        self.fw_model.trainer = self.bw_model.trainer = self.trainer

        with self.datamodule.forward_mode():
            self.fw_model.log = partial(self.sub_log, prefix="fw")
            self.fw_model.print = partial(self.sub_print, prefix="fw")
            self.fw_model.save_predictions = partial(
                _save_prediction, func=self.fw_model.save_predictions, prefix="fw_"
            )
            self.fw_model.save_detailed_predictions = partial(
                _save_detailed_prediction, func=self.fw_model.save_detailed_predictions, prefix="fw_"
            )

        with self.datamodule.backward_mode():
            self.bw_model.log = partial(self.sub_log, prefix="bw")
            self.bw_model.print = partial(self.sub_print, prefix="bw")
            self.bw_model.save_predictions = partial(
                _save_prediction, func=self.bw_model.save_predictions, prefix="bw_"
            )
            self.bw_model.save_detailed_predictions = partial(
                _save_detailed_prediction, func=self.bw_model.save_detailed_predictions, prefix="bw_"
            )

        self.pr_solver = PTAgree()

        if wandb.run is not None:
            tags = []
            for module in [self.fw_model.encoder, self.fw_model.tree_encoder, self.fw_model.decoder]:
                tags.append(type(module).__name__)
            if self.fw_model.embedding is not None:
                tags.append("staticEmb")
            if self.fw_model.pretrained is not None:
                tags.append(self.fw_model.pretrained.name_or_path)
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
                elif "norm" not in name.lower():
                    nn.init.zeros_(param)

    def sub_log(self, name, value, *args, prefix, **kwargs):
        self.log(f"{prefix}/{name}", value, *args, **kwargs)

    def sub_print(self, *args, prefix, **kwargs):
        self.print(prefix, *args, **kwargs)

    def forward(self, batch1, batch2, model1_pred, model2_pred):
        # only contain the code for the agreement constraint
        # only support PCFG
        # reuse the sample in submodel's forward
        # assume PT = [1,1], NT = [2, +\infty]

        if self.current_epoch < self.warmup:
            loss = torch.zeros(1, device=model1_pred["tgt_runtime"]["pred"].device)
        else:
            # if self.reg_method == "pr":
            loss = self.pr_solver(model1_pred["tgt_runtime"]["pred"], model2_pred["tgt_runtime"]["pred"])
            # elif self.reg_method == "emr":
            #     pred1 = model1_pred["tgt_runtime"]["pred"]
            #     pred2 = model2_pred["tgt_runtime"]["pred"]
            #     m_term1, m_trace1 = pred1.dist.marginal_with_grad
            #     m_term2, m_trace2 = pred2.dist.marginal_with_grad
            #     m_term1 = m_term1.view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes).sum(2)
            #     m_term2 = m_term2.view(pred2.batch_size, -1, pred2.pt_states, pred2.pt_num_nodes).sum(2)
            #     m_term2 = m_term2 / (m_term2.sum(2, keepdim=True) + 1e-9)
            #     loss = smoothed_hinge_loss(m_term1 - m_term2.transpose(1, 2), 0.1)

        return {"agreement": loss.mean()}

    def training_step(self, batch: Any, batch_idx: int):
        with self.datamodule.forward_mode():
            out1 = self.fw_model(batch[0])
            loss1 = self.fw_model.training_step(batch[0], batch_idx, forward_prediction=out1)
        with self.datamodule.backward_mode():
            out2 = self.bw_model(batch[1])
            loss2 = self.bw_model.training_step(batch[1], batch_idx, forward_prediction=out2)
        if self.hparams.train_pt_agreement:
            agreement = self(batch[0], batch[1], out1, out2)
            cstrength = self.pt_constraint_strength.get(self.current_epoch)
            self.log("train/agree", agreement["agreement"], prog_bar=True)
            self.log("train/cstrength", cstrength)
        return {"loss": loss1["loss"] + loss2["loss"]}

    def on_validation_epoch_start(self) -> None:
        with self.datamodule.forward_mode():
            self.fw_model.on_validation_epoch_start()
        with self.datamodule.backward_mode():
            self.bw_model.on_validation_epoch_start()

    def validation_step(self, batch: Any, batch_idx: int):
        with self.datamodule.forward_mode():
            output1 = self.fw_model.validation_step(batch[0], batch_idx)
        with self.datamodule.backward_mode():
            output2 = self.bw_model.validation_step(batch[1], batch_idx)
        return {"loss": output1["loss"] + output2["loss"], "output_fw": output1, "output_bw": output2}

    def validation_epoch_end(self, outputs: List[Any]):
        with self.datamodule.forward_mode():
            self.fw_model.validation_epoch_end([item["output_fw"] for item in outputs])
        with self.datamodule.backward_mode():
            self.bw_model.validation_epoch_end([item["output_bw"] for item in outputs])

    @torch.inference_mode(False)
    def on_test_epoch_start(self) -> None:
        with self.datamodule.forward_mode():
            self.fw_model.on_test_epoch_start()
        with self.datamodule.backward_mode():
            self.bw_model.on_test_epoch_start()

    def on_test_epoch_start(self) -> None:
        with self.datamodule.forward_mode():
            self.fw_model.on_test_epoch_start()
        with self.datamodule.backward_mode():
            self.bw_model.on_test_epoch_start()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        with self.datamodule.forward_mode():
            output1 = self.fw_model.test_step(batch[0], batch_idx)
        with self.datamodule.backward_mode():
            output2 = self.bw_model.test_step(batch[1], batch_idx)
        return output1, output2

    def test_epoch_end(self, outputs) -> None:
        with self.datamodule.forward_mode():
            self.fw_model.test_epoch_end([item[0] for item in outputs])
        with self.datamodule.backward_mode():
            self.bw_model.test_epoch_end([item[1] for item in outputs])


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
