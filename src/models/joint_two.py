import logging
import operator
from functools import partial
from types import MethodType
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler
from torch_scatter import scatter_mean
from torch_struct.distributions import SentCFG
from torchmetrics import MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.utils.fn import extract_parses_span_only

from .general_seq2seq import GeneralSeq2SeqModule

log = logging.getLogger(__file__)


class TwoDirectionalModule(ModelBase):
    # model1 is the primary model
    def __init__(self, model1, model2, constraint_strength, optimizer, scheduler):
        super().__init__()
        self.model1: GeneralSeq2SeqModule = instantiate(model1)
        self.model2: GeneralSeq2SeqModule = instantiate(model2)
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)
        assert isinstance(self.trainer.profiler, PassThroughProfiler), "not implemented"
        self.model1.trainer = self.model2.trainer = self.trainer
        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        with self.datamodule.normal_mode():
            self.model1.setup(stage, datamodule)
        self.model1.log = partial(self.sub_log, prefix="m1")
        self.model1.print = partial(self.sub_print, prefix="m1")
        self.model1.save_predictions = partial(
            self.model1.save_predictions, path="m1_predict_on_test.txt"
        )

        with self.datamodule.inverse_mode():
            self.model2.setup(stage, datamodule)
        self.model2.log = partial(self.sub_log, prefix="m2")
        self.model2.print = partial(self.sub_print, prefix="m2")
        self.model2.save_predictions = partial(
            self.model2.save_predictions, path="m2_predict_on_test.txt"
        )

    def sub_log(self, name, value, *args, prefix, **kwargs):
        self.log(f"{prefix}/{name}", value, *args, **kwargs)

    def sub_print(self, *args, prefix, **kwargs):
        self.print(prefix, *args, **kwargs)

    def forward(self, batch, model1_pred, model2_pred):
        # only contain the code for the agreement constraint
        # only support PCFG
        # reuse the sample in submodel's forward
        device = batch["src_ids"].device

        # prepare g(t_1)
        m1t1_event: torch.Tensor = model1_pred["src_runtime"]["event"][-1].sum(-1)
        t1_len = model1_pred["src_runtime"]["dist"].log_potentials[0].shape[1]
        t1_constraint = []
        for offset in range(1, t1_len):
            mask = m1t1_event[:, offset - 1, :-offset] < 0.9
            # mask = m1t1_event.diagonal(offset, dim1=1, dim2=2) < 0.9
            value = torch.full_like(mask, fill_value=-1e9, dtype=torch.float32)
            t1_constraint.append((value, mask))

        # prepare g(t_2). first sample one t_2
        m1t2_params, *_, m1t2_nt_num_nodes = model1_pred["tgt_runtime"]["param"]
        _t, _r, _ro = m1t2_params["term"], m1t2_params["rule"], m1t2_params["root"]
        _c, _l, _a = (
            m1t2_params.get("constraint"),
            m1t2_params.get("lse"),
            m1t2_params.get("add"),
        )
        m1t2_dist = SentCFG((_t, _r, _ro, _c, _l, _a), batch["tgt_lens"])
        m1t2_event = m1t2_dist.argmax[-1].sum(-1)  # TODO sampling
        t2_len = _t.shape[1]
        # scan copy
        # TODO profile and figure out some smart way
        t2_spans = extract_parses_span_only(
            m1t2_dist.argmax[-1], batch["tgt_lens"], inc=1
        )
        for bidx, (spans, l) in enumerate(zip(t2_spans, batch["tgt_lens"])):
            flags = np.zeros((l,), dtype=np.bool8)
            for span in spans:
                if span[1] - span[0] > 1 and not any(flags[span[0] : span[1] + 1]):
                    shape = m1t2_event[
                        bidx, : span[1] - span[0], span[0] : span[1]
                    ].shape
                    m1t2_event[
                        bidx, : span[1] - span[0], span[0] : span[1]
                    ] = torch.flip(torch.triu(torch.ones(shape, device=device)), (1,))
                flags[span[0] : span[1] + 1] = True

        t2_constraint = []
        for offset in range(1, t2_len):
            mask = m1t2_event[:, offset - 1, :-offset] < 0.9
            # mask = m1t2_event.diagonal(offset, dim1=1, dim2=2) < 0.9
            value = torch.full_like(mask, fill_value=-1e9, dtype=torch.float32)
            t2_constraint.append((value, mask))

        # log p(g(t_1) | s_1)
        m1t1_dist = model1_pred["src_runtime"]["dist"]
        potentials = list(m1t1_dist.log_potentials)
        assert len(potentials) == 3
        if len(potentials) < 4:
            potentials.append(None)
        nt = model1_pred["src_runtime"]["event"][1].shape[1]
        m1t1_constraint = []
        for v, m in t1_constraint:
            v = v[..., None].expand(-1, -1, nt)
            m = m[..., None].expand(-1, -1, nt)
            m1t1_constraint.append((v, m))
        potentials[3] = m1t1_constraint
        dist = SentCFG(potentials, batch["src_lens"])
        p_gt1_s1 = dist.partition

        # log p(g(t_2) | g(t_1))
        d1 = self.model1.decoder
        m1t2_cparams = {**m1t2_params}
        nt = d1.nt_states
        m1t2_constraint = []
        for v, m in t2_constraint:
            v = v[..., None, None].expand(-1, -1, nt, m1t2_nt_num_nodes)
            m = m[..., None, None].expand(-1, -1, nt, m1t2_nt_num_nodes)
            m1t2_constraint.append((v, m))
        m1t2_constraint = d1.post_process_nt_constraint(m1t2_constraint, device)
        if "constraint" in m1t2_cparams:
            c = m1t2_cparams["constraint"]
            m1t2_cparams["constraint"] = d1.merge_nt_constraint(c, m1t2_constraint)
        else:
            m1t2_cparams["constraint"] = m1t2_constraint
        p_gt2_gt1 = -d1.pcfg(m1t2_cparams, batch["tgt_lens"])

        # log p(g(t_2) | s_2)
        m2t2_dist = model2_pred["src_runtime"]["dist"]
        potentials = list(m2t2_dist.log_potentials)
        assert len(potentials) == 3
        if len(potentials) < 4:
            potentials.append(None)
        nt = model2_pred["src_runtime"]["event"][1].shape[1]
        m2t2_constraint = []
        for v, m in t2_constraint:
            v = v[..., None].expand(-1, -1, nt)
            m = m[..., None].expand(-1, -1, nt)
            m2t2_constraint.append((v, m))
        potentials[3] = m2t2_constraint
        dist = SentCFG(potentials, batch["tgt_lens"])
        p_gt2_s2 = dist.partition

        # log p(g(t_1) | g(t_2))
        d2 = self.model2.decoder
        m2t1_cparams, *_, m2t1_nt_num_nodes = model2_pred["tgt_runtime"]["param"]
        m2t1_cparams = {**m2t1_cparams}
        nt = d2.nt_states
        m2t1_constraint = []
        for v, m in t1_constraint:
            v = v[..., None, None].expand(-1, -1, nt, m2t1_nt_num_nodes)
            m = m[..., None, None].expand(-1, -1, nt, m2t1_nt_num_nodes)
            m2t1_constraint.append((v, m))
        m2t1_constraint = d2.post_process_nt_constraint(m2t1_constraint, device)
        if "constraint" in m2t1_cparams:
            c = m2t1_cparams["constraint"]
            m2t1_cparams["constraint"] = d2.merge_nt_constraint(c, m2t1_constraint)
        else:
            m2t1_cparams["constraint"] = m2t1_constraint
        p_gt1_gt2 = -d2.pcfg(m2t1_cparams, batch["src_lens"])

        p1 = p_gt1_s1 + p_gt2_gt1
        p2 = p_gt2_s2 + p_gt1_gt2
        logr = p2 - p1
        kl = logr.exp() - 1 - logr
        return {"agreement": kl.mean()}

    def training_step(self, batch: Any, batch_idx: int):
        out1 = self.model1(batch[0])
        out2 = self.model2(batch[1])
        loss1 = self.model1.training_step(batch[0], batch_idx, forward_prediction=out1)
        loss2 = self.model2.training_step(batch[1], batch_idx, forward_prediction=out2)
        agreement = self(batch[0], out1, out2)
        self.log("train/agree", agreement["agreement"], prog_bar=True)
        return {
            "loss": loss1["loss"]
            + loss2["loss"]
            + self.hparams.constraint_strength * agreement["agreement"]
        }

    def on_validation_epoch_start(self) -> None:
        self.model1.on_validation_epoch_start()
        self.model2.on_validation_epoch_start()

    def validation_step(self, batch: Any, batch_idx: int):
        loss1 = self.model1.validation_step(batch[0], batch_idx)["loss"]
        loss2 = self.model2.validation_step(batch[1], batch_idx)["loss"]
        return {"loss": loss1 + loss2}

    def validation_epoch_end(self, outputs: List[Any]):
        self.model1.validation_epoch_end(None)
        self.model2.validation_epoch_end(None)

    def on_test_epoch_start(self) -> None:
        self.model1.on_test_epoch_start()
        self.model2.on_test_epoch_start()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        output1 = self.model1.test_step(batch[0], batch_idx)
        output2 = self.model2.test_step(batch[1], batch_idx)
        return output1, output2

    def test_epoch_end(self, outputs) -> None:
        self.model1.test_epoch_end([item[0] for item in outputs])
        self.model2.test_epoch_end([item[1] for item in outputs])
