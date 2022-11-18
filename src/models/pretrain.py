import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import Metric, MinMetric
from transformers import AutoModel

from src.models.base import ModelBase
from src.models.src_parser.base import SrcParserBase
from src.models.src_parser.gold import GoldTreeProcessor
from src.utils.fn import (
    annotate_snt_with_brackets,
    apply_to_nested_tensor,
    extract_parses,
    get_actions,
    get_tree,
    report_ids_when_err,
)
from src.utils.metric import PerplexityMetric, UnlabeledSpanF1Score

log = logging.getLogger(__file__)


class PretrainPCFGModule(ModelBase):
    def __init__(
        self,
        parser=None,
        optimizer=None,
        scheduler=None,
        load_from_checkpoint=None,
        param_initializer="xavier_uniform",
        domain="src",
    ):
        assert domain in ("src", "tgt")
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        self.parser: SrcParserBase = instantiate(self.hparams.parser, vocab=len(self.datamodule.src_vocab))
        self.gold_tree_processor = GoldTreeProcessor(binarize=False)

        self.train_metric = PerplexityMetric()
        self.ppl_metric = PerplexityMetric()
        self.ppl_best_metric = MinMetric()
        self.uf1_metric = UnlabeledSpanF1Score()

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(self.hparams.load_from_checkpoint, map_location="cpu")
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
                elif "norm" not in name:
                    nn.init.zeros_(param)

    @report_ids_when_err
    def forward(self, batch):
        domain = self.hparams.domain
        ids, lens = batch[f"{domain}_ids"], batch[f"{domain}_lens"]
        dist = self.parser(ids, lens)
        nll = dist.nll
        return {"parser": nll, "nll": nll}

    @report_ids_when_err
    def forward_visualize(self, batch):
        domain = self.hparams.domain
        ids, lens = batch[f"{domain}_ids"], batch[f"{domain}_lens"]

        spans = self.parser.argmax(ids, lens)
        annotated = []
        for snt, span_inst in zip(batch[domain], spans):
            tree = annotate_snt_with_brackets(snt, span_inst)
            annotated.append(tree)

        return {"tree": annotated}

    @report_ids_when_err
    def forward_inference(self, batch):
        domain = self.hparams.domain
        ids, lens = batch[f"{domain}_ids"], batch[f"{domain}_lens"]
        src_spans = self.parser.argmax(ids, lens)
        return {"pred": src_spans}

    def print_prediction(self, batch, batch_size=1):
        training_state = self.training
        self.train(False)
        if isinstance(batch_size, int):
            batch = make_subbatch(batch, batch_size)
        trees = self.forward_visualize(batch)
        for src in trees["tree"]:
            self.print("Tree:", src)
            self.print("=" * 79)
        self.train(training_state)

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["parser"].mean()
        ppl = self.train_metric(output["nll"], batch[f"{self.hparams.domain}_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/parser", output["parser"].mean(), prog_bar=True)

        if batch_idx == 0:
            self.print_prediction(batch, batch_size=1)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.ppl_metric.reset()
        self.uf1_metric.reset()

    @torch.inference_mode(False)
    def validation_step(self, batch: Any, batch_idx: int):
        domain = self.hparams.domain
        batch = apply_to_nested_tensor(batch, func=lambda x: x.clone())
        output = self(batch)
        self.ppl_metric(output["nll"], batch[f"{domain}_lens"])
        if (gtree := batch.get(f"{domain}_tree")) is not None:
            preds = self.forward_inference(batch)["pred"]
            gold_spans = self.gold_tree_processor.get_spans(gtree)
            self.uf1_metric(preds, gold_spans)

        return {"id": batch["id"]}

    def validation_epoch_end(self, outputs: List[Any]):
        f1 = self.uf1_metric.compute()
        ppl = self.ppl_metric.compute()
        self.ppl_best_metric.update(ppl)
        best_ppl = self.ppl_best_metric.compute().item()
        metric = f1 | {"ppl": ppl, "best_ppl": best_ppl}
        self.log_dict({"val/" + k: v for k, v in metric.items()})
        self.print(metric)

    @torch.inference_mode(False)
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.ppl_metric.reset()
        self.uf1_metric.reset()

    @torch.inference_mode(False)
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        f1 = self.uf1_metric.compute()
        ppl = self.ppl_metric.compute()
        metric = f1 | {"ppl": ppl}
        self.log_dict({"test/" + k: v for k, v in metric.items()})
        self.print(metric)


def make_subbatch(batch, size):
    output = {}
    for key, value in batch.items():
        if key == "transformer_inputs":
            output[key] = value
        elif key == "tgt_masks":
            output[key] = [item[:size] for item in value]
        elif key == "src_masks":
            output[key] = [item[:size] for item in value]
        else:
            output[key] = value[:size]
    return output
