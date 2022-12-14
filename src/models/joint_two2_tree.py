import logging
from typing import Any, List, Optional

import torch

from .joint_two2 import TwoDirectionalModule as _TwoDir
from .joint_two2 import smoothed_hinge_loss
from .posterior_regularization.agree import TreeAgree

log = logging.getLogger(__file__)


def smoothed_hinge_loss(d, sigma):
    return torch.where(d.abs() < sigma, d**2 / (2 * sigma), d.abs()).flatten(1).sum(1)
    # return torch.where(d.abs() < sigma, d ** 2 / (2 * sigma), d.abs() - sigma).flatten(1).sum(1)


class TwoDirectionalModule(_TwoDir):
    def __init__(
        self,
        model,
        constraint_strength,
        reg_method,
        optimizer,
        scheduler,
        load_model1_from_checkpoint,
        load_model2_from_checkpoint,
        warmup=0,
    ):
        assert reg_method in ("pr", "emr")
        super().__init__(
            model,
            constraint_strength,
            None,
            optimizer,
            scheduler,
            load_model1_from_checkpoint,
            load_model2_from_checkpoint,
            warmup,
        )

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage, datamodule)
        self.tree_pr_solver = TreeAgree()

    def forward(self, batch1, batch2, model1_pred, model2_pred):
        # only contain the code for the agreement constraint
        # only support PCFG
        # reuse the sample in submodel's forward
        # assume PT = [1,1], NT = [2, +\infty]

        if self.current_epoch < self.warmup:
            loss = torch.zeros(1, device=model1_pred["tgt_runtime"]["pred"].device)
        else:
            if self.reg_method == "pr":
                tgt_dist1 = model1_pred["tgt_runtime"]["pred"].dist
                tgt_dist2 = model2_pred["tgt_runtime"]["pred"].dist
                src_dist1 = model1_pred["src_runtime"]["dist"]
                src_dist2 = model2_pred["src_runtime"]["dist"]

                loss = self.tree_pr_solver(tgt_dist1, src_dist2) + self.tree_pr_solver(tgt_dist2, src_dist1)

            elif self.reg_method == "emr":
                # pred1 = model1_pred["tgt_runtime"]["pred"]
                # pred2 = model2_pred["tgt_runtime"]["pred"]
                # m_term1, m_tgt_trace1 = pred1.dist.marginal_with_grad
                # m_term2, m_tgt_trace2 = pred2.dist.marginal_with_grad
                # m_term1 = m_term1.view(pred1.batch_size, -1, pred1.pt_states, pred1.pt_num_nodes).sum(2)
                # m_term2 = m_term2.view(pred2.batch_size, -1, pred2.pt_states, pred2.pt_num_nodes).sum(2)
                # m_term2 = m_term2 / (m_term2.sum(2, keepdim=True) + 1e-9)
                # token_align_loss = smoothed_hinge_loss(m_term1 - m_term2.transpose(1, 2), 0.1)
                token_align_loss = 0

                _, m_src_trace1 = model1_pred["src_runtime"]["dist"].marginal_with_grad
                _, m_src_trace2 = model2_pred["src_runtime"]["dist"].marginal_with_grad
                m_tgt_trace1 = m_tgt_trace1.flatten(3).sum(3)
                m_tgt_trace2 = m_tgt_trace2.flatten(3).sum(3)
                m_src_trace1 = m_src_trace1.flatten(3).sum(3)
                m_src_trace2 = m_src_trace2.flatten(3).sum(3)

                tree_agreement_loss = smoothed_hinge_loss(m_tgt_trace1 - m_src_trace2, 0.1) + smoothed_hinge_loss(
                    m_tgt_trace2 - m_src_trace1, 0.1
                )

                loss = token_align_loss + tree_agreement_loss

        return {"agreement": loss.mean()}
