from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from src.models.tgt_parser.base import TgtParserPrediction


class RuleConstraintBase:
    def get_mask(
        self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
    ) -> Tensor:
        raise NotImplementedError

    def get_feature(
        self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device
    ) -> Tensor:
        mask = self.get_mask(batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device)
        return 1.0 - mask.float()

    def get_reward(self, batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device):
        mask = self.get_mask(batch_size, pt_states, nt_states, pt_num_nodes, nt_num_nodes, pt_spans, nt_spans, device)
        return mask.float()

    def get_mask_from_pred(self, pred: TgtParserPrediction):
        nt_spans = copy(pred.nt_nodes)
        pt_spans = copy(pred.pt_nodes)
        return self.get_mask(
            pred.batch_size,
            pred.pt_states,
            pred.nt_states,
            pred.pt_num_nodes,
            pred.nt_num_nodes,
            pt_spans,
            nt_spans,
            pred.device,
        )

    def get_feature_from_pred(self, pred: TgtParserPrediction):
        nt_spans = copy(pred.nt_nodes)
        pt_spans = copy(pred.pt_nodes)
        return self.get_feature(
            pred.batch_size,
            pred.pt_states,
            pred.nt_states,
            pred.pt_num_nodes,
            pred.nt_num_nodes,
            pt_spans,
            nt_spans,
            pred.device,
        )

    def get_reward_from_pred(self, pred: TgtParserPrediction):
        nt_spans = copy(pred.nt_nodes)
        pt_spans = copy(pred.pt_nodes)
        return self.get_reward(
            pred.batch_size,
            pred.pt_states,
            pred.nt_states,
            pred.pt_num_nodes,
            pred.nt_num_nodes,
            pt_spans,
            nt_spans,
            pred.device,
        )