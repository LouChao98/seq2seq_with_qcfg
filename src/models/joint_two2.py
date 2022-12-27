import logging
from functools import partial
from typing import Any, List, Optional

import torch
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler

from src.models.base import ModelBase
from src.utils.fn import fix_wandb_tags

from .components.dynamic_hp import DynamicHyperParameter
from .general_seq2seq import GeneralSeq2SeqModule
from .posterior_regularization.agree import NTAlignmentAgree, PTAlignmentAgree, TreeAgree

log = logging.getLogger(__file__)


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


def smoothed_hinge_loss(d, sigma):
    return torch.where(d.abs() < sigma, d**2 / (2 * sigma), d.abs()).flatten(1).sum(1)
    # return torch.where(d.abs() < sigma, d ** 2 / (2 * sigma), d.abs() - sigma).flatten(1).sum(1)


class TwoDirectionalModule(ModelBase):
    # model1 is the primary model
    def __init__(
        self,
        model,
        pt_agreement,
        nt_agreement,
        tree_agreement,
        optimizer,
        scheduler,
        load_model1_from_checkpoint=None,
        load_model2_from_checkpoint=None,
        warmup=0,
    ):
        # pt_agreement, nt_agreement, tree_agreement:
        #   strength: control string for their weights w.r.t. epoch
        #   solver: args to solver
        #   reg_method: control use which solver

        super().__init__()
        self.model1: GeneralSeq2SeqModule = instantiate(model)
        self.model2: GeneralSeq2SeqModule = instantiate(model)
        self.warmup = max(self.model1.hparams.warmup_qcfg, warmup)
        self.save_hyperparameters(logger=False)

        self.pt_alignment_reg_strength = DynamicHyperParameter(pt_agreement.strength)
        self.nt_alignment_reg_strength = DynamicHyperParameter(nt_agreement.strength)
        self.tree_reg_strength = DynamicHyperParameter(tree_agreement.strength)

        self.pr_pt_alignment = PTAlignmentAgree(**pt_agreement.solver)
        self.pr_nt_alignment = NTAlignmentAgree(**nt_agreement.solver)
        self.pr_tree = TreeAgree(**tree_agreement.solver)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)
        assert isinstance(self.trainer.profiler, PassThroughProfiler), "not implemented"
        self.model1.trainer = self.model2.trainer = self.trainer
        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        with self.datamodule.forward_mode():
            self.model1.setup(stage, datamodule)
            self.model1.log = partial(self.sub_log, prefix="m1")
            self.model1.print = partial(self.sub_print, prefix="m1")
            self.model1.save_predictions = partial(_save_prediction, func=self.model1.save_predictions, prefix="m1_")
            self.model1.save_detailed_predictions = partial(
                _save_detailed_prediction, func=self.model1.save_detailed_predictions, prefix="m1_"
            )

        with self.datamodule.backward_mode(), fix_wandb_tags():
            self.model2.setup(stage, datamodule)
            self.model2.log = partial(self.sub_log, prefix="m2")
            self.model2.print = partial(self.sub_print, prefix="m2")
            self.model2.save_predictions = partial(_save_prediction, func=self.model2.save_predictions, prefix="m2_")
            self.model2.save_detailed_predictions = partial(
                _save_detailed_prediction, func=self.model2.save_detailed_predictions, prefix="m2_"
            )

        if self.hparams.load_model1_from_checkpoint is not None:
            state_dict = torch.load(self.hparams.load_model1_from_checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            incompt_params = self.model1.load_state_dict(state_dict, strict=False)
            if len(incompt_params.unexpected_keys) > 0:
                log.warning(f"Unexpected keys: {incompt_params.unexpected_keys}")
            if len(incompt_params.missing_keys) > 0:
                log.warning(f"Missing keys: {incompt_params.missing_keys}")

        if self.hparams.load_model2_from_checkpoint is not None:
            state_dict = torch.load(self.hparams.load_model2_from_checkpoint, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            incompt_params = self.model2.load_state_dict(state_dict, strict=False)
            if len(incompt_params.unexpected_keys) > 0:
                log.warning(f"Unexpected keys: {incompt_params.unexpected_keys}")
            if len(incompt_params.missing_keys) > 0:
                log.warning(f"Missing keys: {incompt_params.missing_keys}")

    def sub_log(self, name, value, *args, prefix, **kwargs):
        self.log(f"{prefix}/{name}", value, *args, **kwargs)

    def sub_print(self, *args, prefix, **kwargs):
        self.print(prefix, *args, **kwargs)

    def forward(self, batch1, batch2, model1_pred, model2_pred):
        # only contain the code for the agreement constraint
        # only support PCFG
        # reuse the sample in submodel's forward
        # assume PT = [1,1], NT = [2, +\infty]

        device = model1_pred["tgt_runtime"]["pred"].device
        logging_vals = {}
        loss = torch.zeros(1, device=device)

        if self.current_epoch < self.warmup:
            return torch.zeros(1, device=device), logging_vals

        pt_agree_weight = self.pt_alignment_reg_strength.get(self.current_epoch)
        logging_vals["pt_agree_weight"] = pt_agree_weight
        if pt_agree_weight > 0:
            _loss = self.get_pt_agreement_loss(batch1, batch2, model1_pred, model2_pred)
            logging_vals["pt_agree_loss"] = _loss
            loss += pt_agree_weight * _loss

        nt_agree_weight = self.nt_alignment_reg_strength.get(self.current_epoch)
        logging_vals["nt_agree_weight"] = nt_agree_weight
        if nt_agree_weight > 0:
            _loss = self.get_nt_agreement_loss(batch1, batch2, model1_pred, model2_pred)
            logging_vals["nt_agree_loss"] = _loss
            loss += nt_agree_weight * _loss

        tree_weight = self.tree_reg_strength.get(self.current_epoch)
        logging_vals["tree_weight"] = tree_weight
        if tree_weight > 0:
            _loss = self.get_tree_agreement_loss(batch1, batch2, model1_pred, model2_pred)
            logging_vals["tree_loss"] = _loss
            loss += tree_weight * _loss

        return loss, logging_vals

    def get_pt_agreement_loss(self, batch1, batch2, model1_pred, model2_pred):
        if self.hparams.pt_agreement.reg_method == "pr":
            loss = self.pr_pt_alignment(model1_pred["tgt_runtime"]["pred"], model2_pred["tgt_runtime"]["pred"])
        elif self.hparams.pt_agreement.reg_method == "emr":
            pred1 = model1_pred["tgt_runtime"]["pred"]
            pred2 = model2_pred["tgt_runtime"]["pred"]
            m_term1, m_trace1 = pred1.dist.marginal_with_grad
            m_term2, m_trace2 = pred2.dist.marginal_with_grad
            m_term1 = m_term1.view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes).sum(2)
            m_term2 = m_term2.view(pred2.batch_size, -1, pred2.pt_states, pred2.pt_num_nodes).sum(2)
            m_term2 = m_term2 / (m_term2.sum(2, keepdim=True) + 1e-9)
            loss = smoothed_hinge_loss(m_term1 - m_term2.transpose(1, 2), 0.1)
        else:
            raise NotImplementedError
        return loss

    def get_nt_agreement_loss(self, batch1, batch2, model1_pred, model2_pred):

        loss = self.pr_nt_alignment(
            model1_pred["tgt_runtime"]["pred"].dist,
            model2_pred["tgt_runtime"]["pred"].dist,
            model1_pred["src_runtime"]["event"]["span"],
            model2_pred["src_runtime"]["event"]["span"],
        )
        return loss

    def get_tree_agreement_loss(self, batch1, batch2, model1_pred, model2_pred):
        if self.hparams.tree_agreement.reg_method == "pr":
            tgt_dist1 = model1_pred["tgt_runtime"]["pred"].dist
            tgt_dist2 = model2_pred["tgt_runtime"]["pred"].dist
            src_dist1 = model1_pred["src_runtime"]["dist"]
            src_dist2 = model2_pred["src_runtime"]["dist"]
            loss = self.pr_tree(tgt_dist1, src_dist2) + self.pr_tree(tgt_dist2, src_dist1)
        elif self.hparams.tree_agreement.reg_method == "emr":
            _, m_src_trace1 = model1_pred["src_runtime"]["dist"].marginal_with_grad
            _, m_src_trace2 = model2_pred["src_runtime"]["dist"].marginal_with_grad
            m_tgt_trace1 = m_tgt_trace1.flatten(3).sum(3)
            m_tgt_trace2 = m_tgt_trace2.flatten(3).sum(3)
            m_src_trace1 = m_src_trace1.flatten(3).sum(3)
            m_src_trace2 = m_src_trace2.flatten(3).sum(3)
            tree_agreement_loss = smoothed_hinge_loss(m_tgt_trace1 - m_src_trace2, 0.1) + smoothed_hinge_loss(
                m_tgt_trace2 - m_src_trace1, 0.1
            )
            loss = tree_agreement_loss
        else:
            raise NotImplementedError
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        with self.datamodule.forward_mode():
            out1 = self.model1(batch[0])
            loss1 = self.model1.training_step(batch[0], batch_idx, forward_prediction=out1)
        with self.datamodule.backward_mode():
            out2 = self.model2(batch[1])
            loss2 = self.model2.training_step(batch[1], batch_idx, forward_prediction=out2)
        loss, logging_vals = self(batch[0], batch[1], out1, out2)
        self.log_dict({"train/" + k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in logging_vals.items()})
        return {"loss": loss1["loss"] + loss2["loss"] + loss.mean()}

    def on_validation_epoch_start(self) -> None:
        with self.datamodule.forward_mode():
            self.model1.on_validation_epoch_start()
        with self.datamodule.backward_mode():
            self.model2.on_validation_epoch_start()

    def validation_step(self, batch: Any, batch_idx: int):
        with self.datamodule.forward_mode():
            output1 = self.model1.validation_step(batch[0], batch_idx)
        with self.datamodule.backward_mode():
            output2 = self.model2.validation_step(batch[1], batch_idx)
        return {"loss": output1["loss"] + output2["loss"], "output1": output1, "output2": output2}

    def validation_epoch_end(self, outputs: List[Any]):
        with self.datamodule.forward_mode():
            self.model1.validation_epoch_end([item["output1"] for item in outputs])
        with self.datamodule.backward_mode():
            self.model2.validation_epoch_end([item["output2"] for item in outputs])

    def on_test_epoch_start(self) -> None:
        with self.datamodule.forward_mode():
            self.model1.on_test_epoch_start()
        with self.datamodule.backward_mode():
            self.model2.on_test_epoch_start()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        with self.datamodule.forward_mode():
            output1 = self.model1.test_step(batch[0], batch_idx)
        with self.datamodule.backward_mode():
            output2 = self.model2.test_step(batch[1], batch_idx)
        return output1, output2

    def test_epoch_end(self, outputs) -> None:
        with self.datamodule.forward_mode():
            self.model1.test_epoch_end([item[0] for item in outputs])
        with self.datamodule.backward_mode():
            self.model2.test_epoch_end([item[1] for item in outputs])
