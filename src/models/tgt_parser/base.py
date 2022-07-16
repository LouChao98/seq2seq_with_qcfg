from typing import List, Optional

import torch.nn as nn

from ...datamodules.components.vocab import VocabularyPair


class TgtParserBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, lengths, node_features, spans, **kwargs):
        raise NotImplementedError

    def parse(self, x, lengths, node_features, spans, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        node_features,
        spans,
        vocab_pair: VocabularyPair,
        src_ids: Optional[List[int]] = None,
    ):
        raise NotImplementedError
