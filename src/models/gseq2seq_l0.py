import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch_lightning.profilers import PassThroughProfiler, SimpleProfiler
from torch_scatter import scatter_mean
from torchmetrics import MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.models.general_seq2seq import GeneralSeq2SeqModule
from src.models.node_filter.base import NodeFilterBase
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


class GSeq2SeqL0Module(GeneralSeq2SeqModule):
    """PR. when decoding, constraint is NOT applied."""

    def __init__(
        self,
        embedding=None,
        transformer_pretrained_model=None,
        encoder=None,
        tree_encoder=None,
        node_filter=None,
        decoder=None,
        parser=None,
        optimizer=None,
        scheduler=None,
        test_metric=None,
        load_from_checkpoint=None,
        param_initializer="xavier_uniform",
        track_param_norm=False,
    ):
        raise NotImplementedError()
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)
        self.profiler = self.trainer.profiler or PassThroughProfiler()

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        self.embedding = instantiate(
            self.hparams.embedding,
            num_embeddings=len(self.datamodule.src_vocab),
        )
        self.pretrained = (
            AutoModel.from_pretrained(self.hparams.transformer_pretrained_model)
            if self.hparams.transformer_pretrained_model is not None
            else None
        )
        self.encoder = instantiate(
            self.hparams.encoder,
            input_dim=0
            + (0 if self.embedding is None else self.embedding.weight.shape[1])
            + (0 if self.pretrained is None else self.pretrained.config.hidden_size),
        )

        self.parser: SrcParserBase = instantiate(
            self.hparams.parser, vocab=len(self.datamodule.src_vocab)
        )
        self.tree_encoder: TreeEncoderBase = instantiate(
            self.hparams.tree_encoder, dim=self.encoder.get_output_dim()
        )
        self.node_filter: NodeFilterBase = instantiate(
            self.hparams.node_filter, dim=self.tree_encoder.get_output_dim()
        )
        self.decoder: TgtParserBase = instantiate(
            self.hparams.decoder,
            vocab=len(self.datamodule.tgt_vocab),
            src_dim=self.tree_encoder.get_output_dim(),
        )

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric = instantiate(self.hparams.test_metric)

        self.setup_patch(stage, datamodule)

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(
                self.hparams.load_from_checkpoint, map_location="cpu"
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict)
        else:
            init_func = {
                "xavier_uniform": nn.init.xavier_uniform_,
                "xavier_normal": nn.init.xavier_normal_,
                "kaiming_uniform": nn.init.kaiming_uniform_,
                "kaiming_normal": nn.init.kaiming_normal_,
            }
            init_func = init_func[self.hparams.param_initializer]
            for name, param in self.named_parameters():
                if param.dim() > 1:
                    init_func(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition
        src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist=dist)
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        gates = self.node_filter(node_spans, node_features, x)
        nf_samples, nf_logprob = self.node_filter.sample(gates)
        node_features, node_spans = self.node_filter.apply_filter(
            nf_samples, node_spans, node_features
        )
        src_logprob = src_logprob + nf_logprob

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
