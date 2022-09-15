import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import MaxMetric, MinMetric

from src.models.base import ModelBase
from src.models.src_parser.base import SrcParserBase
from src.models.src_parser.gold import GoldTreeProcessor
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    get_actions,
    get_tree,
    report_ids_when_err,
)
from src.utils.metric import PerplexityMetric

log = logging.getLogger(__file__)


class SrcParserPretrainModule(ModelBase):
    def __init__(
        self,
        parser=None,
        optimizer=None,
        scheduler=None,
        test_metric=None,
        load_from_checkpoint=None,
        param_initializer="xavier_uniform",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        self.parser: SrcParserBase = instantiate(
            self.hparams.parser, vocab=len(self.datamodule.src_vocab)
        )
        self.gold_tree_processor = GoldTreeProcessor(binarize=False)

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        # self.val_metric = instantiate(self.hparams.test_metric)
        # self.val_best_metric = MaxMetric()
        self.test_metric = instantiate(self.hparams.test_metric)

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

    @report_ids_when_err
    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition

        return {
            "encoder": src_nll.mean(),
            "src_nll": src_nll.mean(),
        }

    @report_ids_when_err
    def forward_visualize(self, batch, sample=False):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]

        parse = self.parser.sample if sample else self.parser.argmax
        src_spans = parse(src_ids, src_lens)[0][-1]
        src_spans, src_trees = extract_parses(src_spans, src_lens, inc=1)
        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))
        return {
            "src_tree": src_annotated,
        }

    @report_ids_when_err
    def forward_inference(self, batch):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]

        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist=dist)[0]
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        return {"src_pred": src_spans}

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["encoder"]
        ppl = self.train_metric(output["src_nll"], batch["src_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/encoder", output["encoder"], prog_bar=True)

        if batch_idx == 0:
            self.eval()
            single_inst = {key: value[:2] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src in trees["src_tree"]:
                self.print("Src:", src)
            self.train()
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_metric.reset()

    @torch.inference_mode(False)
    def validation_step(self, batch: Any, batch_idx: int):
        batch["src_ids"] = batch["src_ids"].clone()

        output = self(batch)
        loss = output["encoder"]

        # preds = self.forward_inference(batch)["src_pred"]
        # gold_spans = self.gold_tree_processor.get_spans(batch)
        # self.val_metric(preds, gold_spans)
        self.val_metric(output["src_nll"], batch["src_lens"])
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        # f1 = self.val_metric.compute()['f1']  # get val accuracy from current epoch
        # self.val_best_metric.update(f1)
        ppl = self.val_metric.compute()
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute()
        # self.log("val/f1", f1, on_epoch=True, prog_bar=True)
        # self.log("val/f1_best", best_ppl, on_epoch=True, prog_bar=True)
        # self.print("val/f1", str(f1.item()))
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)
        self.print("val/ppl", str(ppl.item()))

    @torch.inference_mode(False)
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        # patch for inference_mode
        batch["src_ids"] = batch["src_ids"].clone()

        preds = self.forward_inference(batch)["src_pred"]
        gold_spans = self.gold_tree_processor.get_spans(batch)

        self.test_metric(preds, gold_spans)

        return {"preds": preds, "targets": gold_spans, "id": batch["id"]}

    def test_epoch_end(self, outputs) -> None:
        acc = self.test_metric.compute()
        self.log_dict({"test/" + k: v for k, v in acc.items()})
        self.print(acc)
        # self.save_predictions(outputs)
