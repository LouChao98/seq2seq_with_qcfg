import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch_scatter import scatter_mean
from torchmetrics import MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.models.general_seq2seq import GeneralSeq2SeqModule
from src.models.src_parser.base import SrcParserBase
from src.models.tgt_parser.base import TgtParserBase
from src.models.tree_encoder.base import TreeEncoderBase
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    get_actions,
    get_tree,
)
from src.utils.metric import PerplexityMetric

log = logging.getLogger(__file__)


class GSeq2SeqPosterioirRegularizationModule(GeneralSeq2SeqModule):
    """PR. when decoding, constraint is NOT applied."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition
        src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist=dist)
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_params, *_ = self.decoder.get_params(
            node_features, node_spans, batch["tgt_ids"], copy_position=copy_position
        )
        tgt_nll = self.decoder(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            params=tgt_params,
            copy_position=copy_position,
        )
        tgt_pr = self.decoder.forward_pr(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            params=tgt_params,
            copy_position=copy_position,
        )

        with torch.no_grad():
            src_spans_argmax, src_logprob_argmax = self.parser.argmax(
                src_ids, src_lens, dist=dist
            )
            src_spans_argmax = extract_parses(src_spans_argmax[-1], src_lens, inc=1)[0]
            node_features_argmax, node_spans_argmax = self.tree_encoder(
                x, src_lens, spans=src_spans_argmax
            )
            tgt_nll_argmax = self.decoder(
                batch["tgt_ids"],
                batch["tgt_lens"],
                node_features_argmax,
                node_spans_argmax,
                copy_position=copy_position,
            )
            neg_reward = (tgt_nll - tgt_nll_argmax).detach()

        return {
            "decoder": tgt_nll.mean() + tgt_pr.mean(),
            "encoder": src_nll.mean() + (src_logprob * neg_reward).mean(),
            "tgt_nll": tgt_nll.mean(),
            "src_nll": src_nll.mean(),
            "reward": -neg_reward.mean(),
        }

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/decoder", output["decoder"], prog_bar=True)
        self.log("train/encoder", output["encoder"], prog_bar=True)
        if "reward" in output:
            self.log("train/reward", output["reward"])

        if batch_idx == 0:
            self.eval()
            single_inst = {key: value[:2] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(
                trees["src_tree"], trees["tgt_tree"], trees["alignment"]
            ):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print(
                    "Alg:\n"
                    + "\n".join(map(lambda x: f"  {x[0]} - {x[1]} {x[2]}", alg))
                )
            self.train()
        return {"loss": loss}
