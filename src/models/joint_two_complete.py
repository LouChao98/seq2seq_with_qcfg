import logging
from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
import torch_struct
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler
from torch_struct.distributions import SentCFG

from src.models.base import ModelBase
from src.utils.fn import apply_to_nested_tensor, extract_parses_span_only, fix_wandb_tags

from .components.dynamic_hp import DynamicHyperParameter
from .general_seq2seq import GeneralSeq2SeqModule

log = logging.getLogger(__file__)


class TwoDirectionalCompleteModule(ModelBase):
    # model1 is the primary model
    def __init__(
        self,
        model1,
        model2,
        constraint_strength,
        constraint_estimation_strategy,
        optimizer,
        scheduler,
    ):
        assert constraint_estimation_strategy in ("sample", "argmax")
        super().__init__()
        self.model1: GeneralSeq2SeqModule = instantiate(model1)
        self.model2: GeneralSeq2SeqModule = instantiate(model2)
        self.save_hyperparameters(logger=False)

        self.constraint_strength = DynamicHyperParameter(constraint_strength)

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
        self.model1.save_predictions = partial(self.model1.save_predictions, path="m1_predict_on_test.txt")

        with self.datamodule.inverse_mode(), fix_wandb_tags():
            self.model2.setup(stage, datamodule)
        self.model2.log = partial(self.sub_log, prefix="m2")
        self.model2.print = partial(self.sub_print, prefix="m2")
        self.model2.save_predictions = partial(self.model2.save_predictions, path="m2_predict_on_test.txt")

    def sub_log(self, name, value, *args, prefix, **kwargs):
        self.log(f"{prefix}/{name}", value, *args, **kwargs)

    def sub_print(self, *args, prefix, **kwargs):
        self.print(prefix, *args, **kwargs)

    def forward(self, batch1, batch2, model1_pred, model2_pred):
        # only contain the code for the agreement constraint
        # only support PCFG
        # reuse the sample in submodel's forward
        # assume PT = [1,1], NT = [2, +\infty]
        device = batch1["src_ids"].device
        bsz = len(batch1["src_lens"])
        event_key = "event" if self.hparams.constraint_estimation_strategy == "sample" else "argmax_event"
        m1nt = self.model1.decoder.nt_states
        m2nt = self.model2.decoder.nt_states

        # prepare g(t_1)
        m1t1_event: torch.Tensor = model1_pred["src_runtime"][event_key][-1].sum(-1)
        t1_len = model1_pred["src_runtime"]["dist"].log_potentials[0].shape[1]
        t1_constraint = self.get_constraint_list_from_event(m1t1_event, t1_len)

        # prepare g(t_2). first sample one t_2
        m1t2_params = model1_pred["tgt_runtime"]["param"][0]
        m1t2_dist = SentCFG.from_dict(m1t2_params, batch1["tgt_lens"])
        if self.hparams.constraint_estimation_strategy == "sample":
            m1t2_event_term, *_, m1t2_event_labeled = m1t2_dist._struct(torch_struct.SampledSemiring).marginals(
                m1t2_dist.log_potentials, lengths=m1t2_dist.lengths
            )
        else:
            m1t2_event_term, *_, m1t2_event_labeled = m1t2_dist.argmax
        t2_len = m1t2_dist.log_potentials[0].shape[1]
        m1t2_event_term = m1t2_event_term.argmax(2).tolist()
        if self.model1.decoder.nt_states == self.model1.decoder.pt_states:
            m1t2_event_a = m1t2_event_labeled.view(bsz, t2_len, t2_len, self.model1.decoder.nt_states, -1).sum(-2)
        else:
            raise NotImplementedError
        m1t2_event = m1t2_event_labeled.sum(-1)
        # scan copy
        t2_spans = extract_parses_span_only(m1t2_event_labeled, batch1["tgt_lens"], inc=1)
        for bidx, (spans, l) in enumerate(zip(t2_spans, batch1["tgt_lens"])):
            flags = np.zeros((l,), dtype=np.bool8)
            for i, span in enumerate(spans):
                if span[1] - span[0] > 1 and not any(flags[span[0] : span[1] + 1]):
                    loc = (bidx, slice(span[1] - span[0]), slice(span[0], span[1]))
                    shape = m1t2_event[loc].shape
                    patch = torch.fliplr(torch.triu(torch.ones(shape, device=device)))
                    m1t2_event[loc] = patch
                    m1t2_event_a[loc] = patch[..., None]
                flags[span[0] : span[1] + 1] = True
        t2_constraint = self.get_constraint_list_from_event(m1t2_event, t2_len)
        t2_constraint_a = self.get_constraint_list_from_event(m1t2_event_a, t2_len)
        t2_constraint_a = apply_to_nested_tensor(
            t2_constraint_a,
            lambda x: x.unsqueeze(2).expand(-1, -1, m1nt, -1),
        )

        # log p(g(t_1) | s_1)
        m1t1_dist = model1_pred["src_runtime"]["dist"]
        potentials = list(m1t1_dist.log_potentials)
        assert len(potentials) == 3
        if len(potentials) < 4:
            potentials.append(None)
        potentials[3] = self.expand_constraint(t1_constraint, -1, m1nt)
        dist = SentCFG(potentials, batch1["src_lens"])
        p_gt1_s1 = dist.partition

        # log p(g(t_2) | g(t_1))
        d1 = self.model1.decoder
        m1t2_cparams = {**m1t2_params}
        m1t2_constraint = d1.post_process_nt_constraint(t2_constraint_a, device)
        if "constraint" in m1t2_cparams:
            c = m1t2_cparams["constraint"]
            m1t2_cparams["constraint"] = d1.merge_nt_constraint(c, m1t2_constraint)
        else:
            m1t2_cparams["constraint"] = m1t2_constraint
        p_gt2_gt1 = -d1.pcfg(m1t2_cparams, batch1["tgt_lens"])

        # log p(g(t_2) | s_2)
        m2t2_dist = model2_pred["src_runtime"]["dist"]
        potentials = list(m2t2_dist.log_potentials)
        assert len(potentials) == 3
        if len(potentials) < 4:
            potentials.append(None)
        potentials[3] = self.expand_constraint(t2_constraint, -1, m2nt)
        dist = SentCFG(potentials, batch2["src_lens"])
        p_gt2_s2 = dist.partition

        # log p(g(t_1) | g(t_2))
        node_features, node_spans = self.model2.tree_encoder(
            model2_pred["runtime"]["seq_encoded"], batch2["src_lens"], spans=t2_spans
        )
        copy_position = (batch2.get("copy_token"), batch2.get("copy_phrase"))
        m2_tgt_params = self.model2.decoder.get_params(
            node_features, node_spans, batch2["tgt_ids"], copy_position=copy_position
        )[0]
        m1t1_pt_spans = model1_pred["tgt_runtime"]["param"][1]
        m1t1_pt_num_spans = model1_pred["tgt_runtime"]["param"][2]
        m1t1_nt_spans = model1_pred["tgt_runtime"]["param"][3]
        m1t1_nt_num_spans = model1_pred["tgt_runtime"]["param"][4]
        m2t2_pt_num_spans = model2_pred["tgt_runtime"]["param"][2]
        m2t2_nt_num_spans = model2_pred["tgt_runtime"]["param"][4]
        inv_nt_align = torch.zeros((bsz, t1_len, t1_len, m2t2_nt_num_spans))
        inv_pt_align = torch.zeros((bsz, t1_len, m2t2_pt_num_spans))
        spans_label = m1t2_event_labeled.argmax(-1).cpu().numpy()
        for bidx, spans in enumerate(node_spans):
            nt_i = 0
            for i, span in enumerate(spans):
                if span[0] < span[1]:
                    label = spans_label[bidx, span[1] - span[0] - 1, span[0]]
                    s = label % m1t1_nt_num_spans
                    s = m1t1_nt_spans[bidx][s]
                    inv_nt_align[bidx, s[1] - s[0] - 1, s[0], nt_i] = 1
                    nt_i += 1
                else:
                    label = m1t2_event_term[bidx][span[0]]
                    s = label % m1t1_pt_num_spans
                    s = m1t1_pt_spans[bidx][s]
                    inv_pt_align[bidx, s[0], i] = 1
        inv_nt_align = inv_nt_align.to(device)
        inv_pt_align = inv_pt_align.to(device)
        d2 = self.model2.decoder
        m2t1_constraint = self.get_constraint_list_from_event(inv_nt_align, t1_len)
        m2t1_constraint = self.expand_constraint(m2t1_constraint, 2, m2nt)
        m2t1_constraint = d2.post_process_nt_constraint(m2t1_constraint, device)
        if "constraint" in m2_tgt_params:
            c = m2_tgt_params["constraint"]
            m2_tgt_params["constraint"] = d2.merge_nt_constraint(c, m2t1_constraint)
        else:
            m2_tgt_params["constraint"] = m2t1_constraint
        p_gt1_gt2 = -d2(
            batch2["tgt_ids"],
            batch2["tgt_lens"],
            node_features,
            node_spans,
            params=m2_tgt_params,
            copy_position=copy_position,
        )

        p1 = p_gt1_s1 + p_gt2_gt1
        p2 = p_gt2_s2 + p_gt1_gt2
        logr = p2 - p1
        kl = logr.exp() - 1 - logr
        kl = kl.clamp(max=100)
        return {"agreement": kl.mean()}

    def get_constraint_list_from_event(self, event, length):
        constraint = []
        for offset in range(1, length):
            mask = event[:, offset - 1, :-offset] < 0.9
            # mask = m1t1_event.diagonal(offset, dim1=1, dim2=2) < 0.9
            value = torch.full_like(mask, fill_value=-1e9, dtype=torch.float32)
            constraint.append((value, mask))
        return constraint

    def expand_constraint(self, constraint, dim, size):
        return apply_to_nested_tensor(constraint, partial(self.expand_tensor, dim=dim, size=size))

    def expand_tensor(self, tensor, dim, size):
        t = [-1] * (tensor.ndim + 1)
        t[dim] = size
        return tensor.unsqueeze(dim).expand(*t)

    def training_step(self, batch: Any, batch_idx: int):
        out1 = self.model1(batch[0])
        out2 = self.model2(batch[1])
        loss1 = self.model1.training_step(batch[0], batch_idx, forward_prediction=out1)
        loss2 = self.model2.training_step(batch[1], batch_idx, forward_prediction=out2)
        agreement = self(batch[0], batch[1], out1, out2)
        self.log("train/agree", agreement["agreement"], prog_bar=True)
        cstrength = self.constraint_strength.get(self.current_epoch)
        return {"loss": loss1["loss"] + loss2["loss"] + cstrength * agreement["agreement"]}

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
