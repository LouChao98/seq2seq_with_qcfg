import logging
from functools import partial
from typing import Any, List, Optional

import torch
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler

from src.models.base import ModelBase

from .components.dynamic_hp import DynamicHyperParameter
from .general_seq2seq import GeneralSeq2SeqModule
from .posterior_regularization.agree import PTAgree

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


class TwoDirectionalModule(ModelBase):
    # model1 is the primary model
    def __init__(
        self,
        model1,
        model2,
        constraint_strength,
        optimizer,
        scheduler,
    ):
        super().__init__()
        self.model1: GeneralSeq2SeqModule = instantiate(model1)
        self.model2: GeneralSeq2SeqModule = instantiate(model2)
        self.warmup = self.model1.hparams.warmup_qcfg
        self.save_hyperparameters(logger=False)

        self.constraint_strength = DynamicHyperParameter(constraint_strength)

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

        with self.datamodule.backward_mode():
            self.model2.setup(stage, datamodule)
            self.model2.log = partial(self.sub_log, prefix="m2")
            self.model2.print = partial(self.sub_print, prefix="m2")
            self.model2.save_predictions = partial(_save_prediction, func=self.model2.save_predictions, prefix="m2_")
            self.model2.save_detailed_predictions = partial(
                _save_detailed_prediction, func=self.model2.save_detailed_predictions, prefix="m2_"
            )

        self.pr_solver = PTAgree()

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
            loss = self.pr_solver(model1_pred["tgt_runtime"]["pred"], model2_pred["tgt_runtime"]["pred"])

        return {"agreement": loss.mean()}

    def training_step(self, batch: Any, batch_idx: int):
        with self.datamodule.forward_mode():
            out1 = self.model1(batch[0])
            loss1 = self.model1.training_step(batch[0], batch_idx, forward_prediction=out1)
        with self.datamodule.backward_mode():
            out2 = self.model2(batch[1])
            loss2 = self.model2.training_step(batch[1], batch_idx, forward_prediction=out2)
        agreement = self(batch[0], batch[1], out1, out2)
        cstrength = self.constraint_strength.get(self.current_epoch)
        self.log("train/agree", agreement["agreement"], prog_bar=True)
        self.log("train/cstrength", cstrength)
        return {"loss": loss1["loss"] + loss2["loss"] + cstrength * agreement["agreement"]}

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
