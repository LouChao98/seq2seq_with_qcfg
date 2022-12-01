import torch
import torch.nn as nn

from .base import NodeFilterBase


class NodeFilterV1(NodeFilterBase):
    def __init__(self, dim, hidden) -> None:
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, spans, span_features, seq_features, **kwargs):
        gates = []
        for feats, span in zip(span_features, spans):
            gate = self.nn(feats).squeeze(-1).clamp(0.2)
            mask = [l == r for l, r, _ in span]
            mask[-1] = True
            mask = torch.tensor(mask, dtype=torch.float, device=gate.device)
            gate = torch.max(mask, gate)
            gates.append(gate)
        return gates
