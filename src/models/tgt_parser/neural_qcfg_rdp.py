from typing import List

import torch

from ...datamodules.components.vocab import VocabularyPair
from .neural_qcfg import NeuralQCFGTgtParser
from .struct.pcfg import PCFG
from .struct.pcfg_rdp import PCFGRandomizedDP


class NeuralQCFGRDPTgtParser(NeuralQCFGTgtParser):
    def __init__(
        self, rdp_topk=5, rdp_sample_size=5, rdp_smooth=0.001, *args, **kwargs
    ):
        super(NeuralQCFGRDPTgtParser, self).__init__(*args, **kwargs)
        self.neg_huge = -1e5
        self.pcfg_inside = PCFGRandomizedDP(rdp_topk, rdp_sample_size, rdp_smooth)
        self.pcfg_decode = PCFG()

    def forward(self, x, lengths, node_features, spans, copy_position=None):
        self.pcfg = self.pcfg_inside
        return super().forward(x, lengths, node_features, spans, copy_position)

    def parse(self, x, lengths, node_features, spans, copy_position=None):
        # self.pcfg = self.pcfg_decode
        self.pcfg = self.pcfg_inside
        return super().parse(x, lengths, node_features, spans, copy_position)

    def generate(
        self,
        node_features,
        spans,
        vocab_pair: VocabularyPair,
        src_ids: torch.Tensor,
        src: List[List[str]],
    ):
        self.pcfg = self.pcfg_decode
        return super().generate(node_features, spans, vocab_pair, src_ids, src)

