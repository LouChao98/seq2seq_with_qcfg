import logging
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import MinMetric

from src.models.base import ModelBase
from src.models.src_parser.base import SrcParserBase
from src.models.tgt_parser.base import TgtParserBase
from src.models.tree_encoder.base import TreeEncoderBase
from src.utils.fn import annotate_snt_with_brackets, extract_parses, get_actions, get_tree
from src.utils.metric import PerplexityMetric, WholeSentMatchAccuracy

log = logging.getLogger(__file__)
DEBUG = False

class ScanModule(ModelBase):
    def __init__(
        self,
        embedding_dim=None,
        tree_encoder=None,
        decoder=None,
        parser=None,
        optimizer=None,
        scheduler=None,
        load_from_checkpoint=None,
        loss_threshold=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)
        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule
        self.embedding = nn.Embedding(
            len(self.datamodule.src_vocab), self.hparams.embedding_dim
        )
        self.parser: SrcParserBase = instantiate(
            self.hparams.parser, vocab=len(self.datamodule.src_vocab)
        )
        self.tree_encoder: TreeEncoderBase = instantiate(self.hparams.tree_encoder)
        self.decoder: TgtParserBase = instantiate(
            self.hparams.decoder, vocab=len(self.datamodule.tgt_vocab)
        )

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric = WholeSentMatchAccuracy()

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(
                self.hparams.load_from_checkpoint, map_location="cpu"
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict)
        else:
            for name, param in self.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, batch, _debug=DEBUG):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x: torch.Tensor = self.embedding(src_ids)

        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition

        if _debug:
            for i in range(len(src_ids)):
                _ref_lens = src_lens[i: i+1]
                _ref_ids = src_ids[i: i+1, :_ref_lens[0]]
                _ref_dist = self.parser(_ref_ids, _ref_lens)
                assert torch.isclose(src_nll[i: i+1], -_ref_dist.partition)

        src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist)
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)
        tgt_nll = self.decoder(
            batch["tgt_ids"], batch["tgt_lens"], node_features, node_spans,
        )

        if _debug:
            for i in range(len(src_ids)):
                _ref_node_features, _ref_node_spans = self.tree_encoder(x[i:i+1, :src_lens[i]], src_lens[i:i+1], spans=src_spans[i:i+1])
                assert all(map(lambda x:torch.allclose(*x), zip(node_features[i:i+1], _ref_node_features)))
                assert node_spans[i:i+1] == _ref_node_spans
                _ref_nll = self.decoder(
                    batch["tgt_ids"][i:i+1, :batch["tgt_lens"][i]],
                    batch["tgt_lens"][i:i+1],
                    _ref_node_features,
                    _ref_node_spans,
                )
                assert torch.isclose(tgt_nll[i:i+1], _ref_nll)

        with torch.no_grad():
            src_spans_argmax, src_logprob_argmax = self.parser.argmax(
                src_ids, src_lens, dist
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
            )
            neg_reward = (tgt_nll - tgt_nll_argmax).detach()

            if _debug:
                for i in range(len(src_ids)):
                    _ref_lens = src_lens[i: i+1]
                    _ref_ids = src_ids[i: i+1, :_ref_lens[0]]
                    _ref_dist = self.parser(_ref_ids, _ref_lens)
                    _ref_spans_argmax, _ref_logprob_argmax = self.parser.argmax(
                        _ref_ids, _ref_lens, _ref_dist
                    )
                    _ref_spans_argmax = extract_parses(_ref_spans_argmax[-1], _ref_lens, inc=1)[0]
                    assert src_spans_argmax[i] == _ref_spans_argmax[0]
                    assert torch.allclose(src_logprob_argmax[i], _ref_logprob_argmax[0])

        output = {
            "decoder": tgt_nll.mean(),
            "encoder": src_nll.mean() + (src_logprob * neg_reward).mean(),
            "tgt_nll": tgt_nll.sum(),
            "src_nll": src_nll.sum(),
            "reward": -neg_reward,
        }

        if _debug:
            self.optimizers().zero_grad()
            (output['decoder'] + output['encoder']).backward(retain_graph=True)
            grads = {name: param.grad.clone() for name, param in self.named_parameters()}
            self.optimizers().zero_grad()
            for i in range(len(src_ids) - 1, -1, -1):
                (tgt_nll[i] + src_nll[i] + src_logprob[i] * neg_reward[i]).backward(retain_graph=True)
            for name, param in self.named_parameters():
                assert torch.allclose(param.grad / len(src_ids), grads[name], atol=1e-6)

        if self.hparams.loss_threshold is not None:
            output['decoder'].clamp(min=self.hparams.loss_threshold)

        return output

    def forward_visualize(self, batch, sample=False):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)

        parse = self.parser.sample if sample else self.parser.argmax
        src_spans = parse(src_ids, src_lens)[0][-1]
        src_spans, src_trees = extract_parses(src_spans, src_lens, inc=1)
        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))

        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_spans, aligned_spans = self.decoder.parse(
            batch["tgt_ids"], batch["tgt_lens"], node_features, node_spans,
        )
        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)
        alignments = []
        for tgt_spans_inst, tgt_snt, aligned_spans_inst, src_snt in zip(
            tgt_spans, batch["tgt"], aligned_spans, batch["src"]
        ):
            alignments_inst = []
            for tgt_span, src_span in zip(tgt_spans_inst, aligned_spans_inst):
                alignments_inst.append(
                    (
                        " ".join(src_snt[src_span[0] : src_span[1] + 1])
                        + f" ({src_span[0]}, {src_span[1]+1})",
                        " ".join(tgt_snt[tgt_span[0] : tgt_span[1] + 1])
                        + f" ({tgt_span[0]}, {tgt_span[1]+1})",
                    )
                )
            alignments.append(alignments_inst)
        return {
            "src_tree": src_annotated,
            "tgt_tree": tgt_annotated,
            "alignment": alignments,
        }

    def forward_inference(self, batch):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)
        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist)[0]
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        y_preds = self.decoder.generate(
            node_features, node_spans, self.datamodule.tgt_vocab, batch['src']
        )

        # tgt_nll = self.decoder(
        #     batch["tgt_ids"],
        #     batch["tgt_lens"],
        #     node_features,
        #     node_spans,
        # )
        # tgt_ppl = np.exp(tgt_nll.detach().cpu().numpy() / batch["tgt_lens"])

        return {"pred": [item[0] for item in y_preds]}

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/decoder", output["decoder"], prog_bar=True)
        self.log("train/encoder", output["encoder"], prog_bar=True)
        self.log("train/reward", output["reward"])

        if batch_idx == 0:
            single_inst = {key: value[:2] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(
                trees["src_tree"], trees["tgt_tree"], trees["alignment"]
            ):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print(
                    "Alg:\n" + "\n".join(map(lambda x: f"  {x[0]}, {x[1]}", alg))
                )
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute()
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)
        self.print("val/ppl", str(ppl.item()))

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        preds = self.forward_inference(batch)["pred"]
        targets = batch["tgt"]

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        if batch_idx == 0:
            single_inst = {key: value[:2] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            self.print("=" * 79)
            for src, tgt, alg in zip(
                trees["src_tree"], trees["tgt_tree"], trees["alignment"]
            ):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
                self.print(
                    "Alg:\n" + "\n".join(map(lambda x: f"  {x[0]}, {x[1]}", alg))
                )

        return {"preds": preds, "targets": targets, "id": batch["id"]}

    def test_epoch_end(self, outputs) -> None:
        self.print("test/acc", str(self.test_metric.compute().item()))
        if self.global_rank == 0:
            # TODO check whether pl gather outputs for me
            preds = []
            for inst in outputs:
                preds_batch = inst["preds"]
                id_batch = inst["id"].tolist()
                preds.extend(zip(id_batch, preds_batch))
            preds.sort(key=lambda x: x[0])

            with open("predict_on_test.txt", "w") as f:
                for _, inst in preds:
                    f.write(" ".join(inst))
                    f.write("\n")
