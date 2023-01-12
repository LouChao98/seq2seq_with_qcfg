import logging
import math
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Counter, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize

from src.datamodules.datamodule import _DataModule
from src.utils.fn import apply_to_nested_tensor

from ..components.clamp import uni_dir_differentiable_clamp
from ..constraint.base import RuleConstraintBase
from ..struct.base import DecompBase, DecompSamplerBase, TokenType
from .base import DirTgtParserPrediction, TgtParserBase, TgtParserPrediction

logger = logging.getLogger(__file__)


class TgtStruattParserBase(TgtParserBase):
    def forward(self, node_features, spans, weights, **kwargs) -> TgtParserPrediction:
        raise NotImplementedError

    def build_src_features(self, spans, node_features, weights):
        # seperate nt and pt features according to span width

        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        pt_weights, nt_weights = [], []
        for spans_item, node_features_item, weights_item in zip(spans, node_features, weights):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            pt_weight = []
            nt_weight = []
            for s, f, w in zip(spans_item, node_features_item, weights_item):
                s_len = s[1] - s[0]
                if self.nt_span_range[0] <= s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(f)
                    nt_span.append(s)
                    nt_weight.append(w)
                if self.pt_span_range[0] <= s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(f)
                    pt_span.append(s)
                    pt_weight.append(w)
            self.sanity_check_spans(nt_span, pt_span)
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
            pt_weights.append(torch.stack(pt_weight))
            nt_weights.append(torch.stack(nt_weight))
        nt_num_nodes_list = [len(item) for item in nt_node_features]
        pt_num_nodes_list = [len(item) for item in pt_node_features]
        nt_node_features = pad_sequence(nt_node_features, batch_first=True, padding_value=0.0)
        pt_node_features = pad_sequence(pt_node_features, batch_first=True, padding_value=0.0)
        nt_weights = pad_sequence(nt_weights, batch_first=True, padding_value=0.0).clamp(1e-32).log()
        pt_weights = pad_sequence(pt_weights, batch_first=True, padding_value=0.0).clamp(1e-32).log()
        # if self.ignore_weight:
        #     nt_weights.zero_()
        #     pt_weights.zero_()
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)

        return (
            nt_spans,
            nt_num_nodes_list,
            nt_num_nodes,
            nt_node_features,
            nt_weights,
            pt_spans,
            pt_num_nodes_list,
            pt_num_nodes,
            pt_node_features,
            pt_weights,
        )
