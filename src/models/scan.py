import logging
import re
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_struct
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence
from torch_struct import SentCFG
from torchmetrics import MinMetric
from src.datamodules.components.vocab import Vocabulary

from src.models.components.common import MultiResidualLayer
from src.models.components.pcfg import PCFG, FastestTDPCFG
from src.utils.fn import (
    annotate_snt_with_brackets,
    extract_parses,
    get_actions,
    get_tree,
)
from src.utils.metric import PerplexityMetric, WholeSentMatchAccuracy

log = logging.getLogger(__file__)


class ScanModule(pl.LightningModule):
    def __init__(
        self,
        embedding_dim=None,
        tree_encoder=None,
        decoder=None,
        parser=None,
        optimizer=None,
        scheduler=None,
        load_from_checkpoint=None,
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
        self.tree_encoder = instantiate(self.hparams.tree_encoder)
        self.decoder: NeuralQCFGTgtParser = instantiate(
            self.hparams.decoder, vocab=len(self.datamodule.tgt_vocab)
        )
        self.parser: NeuralPCFGSrcParser = instantiate(
            self.hparams.parser, vocab=len(self.datamodule.src_vocab)
        )

        self.train_metric = PerplexityMetric()
        self.val_metric = PerplexityMetric()
        self.val_best_metric = MinMetric()
        self.test_metric = WholeSentMatchAccuracy()

        if self.hparams.load_from_checkpoint is not None:
            state_dict = torch.load(
                self.hparams.load_from_checkpoint, map_location="cpu"
            )["state_dict"]
            self.load_state_dict(state_dict)

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)
        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition

        src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist)
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)
        tgt_nll = self.decoder(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            argmax=False,
        )

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
                argmax=False,
            )
            neg_reward = (tgt_nll - tgt_nll_argmax).detach()

        # if tgt_nll.mean().abs() > 1e5 or (src_logprob * neg_reward).mean().abs() > 1e5:
        #     breakpoint()
        return {
            "decoder": tgt_nll.mean(),
            "encoder": (src_logprob * neg_reward).mean(),
            "tgt_nll": tgt_nll.sum(),
            "src_nll": src_nll.sum(),
        }

    def forward_visualize(self, batch):
        # parse and annotate brackets on src and tgt
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)
        dist = self.parser(src_ids, src_lens)

        src_spans = self.parser.argmax(src_ids, src_lens, dist)[0][-1]
        src_spans, src_trees = extract_parses(src_spans, src_lens, inc=1)
        src_actions, src_annotated = [], []
        for snt, tree in zip(batch["src"], src_trees):
            src_actions.append(get_actions(tree))
            src_annotated.append(get_tree(src_actions[-1], snt))
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)
        tgt_spans = self.decoder(
            batch["tgt_ids"], batch["tgt_lens"], node_features, node_spans, argmax=True,
        )[0]
        tgt_annotated = []
        for snt, tgt_spans_inst in zip(batch["tgt"], tgt_spans):
            tree = annotate_snt_with_brackets(snt, tgt_spans_inst)
            tgt_annotated.append(tree)
        return {
            "src_tree": src_annotated,
            "tgt_tree": tgt_annotated,
        }

    def forward_inference(self, batch):
        # actually predict the target sequence
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        x = self.embedding(src_ids)
        dist = self.parser(src_ids, src_lens)
        src_spans = self.parser.argmax(src_ids, src_lens, dist)[0]
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        y_preds = self.decoder.decode(
            node_features, node_spans, self.datamodule.tgt_vocab
        )
        # tgt_nll = self.decoder(
        #     batch["tgt_ids"],
        #     batch["tgt_lens"],
        #     node_features,
        #     node_spans,
        #     argmax=False,
        # )

        return {"pred": y_preds}

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        ppl = self.train_metric(output["tgt_nll"], batch["tgt_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train/decoder", output['decoder'], prog_bar=True)
        self.log("train/encoder", output['encoder'], prog_bar=True)

        if batch_idx == 0:
            single_inst = {key: value[:10] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            for src, tgt in zip(trees["src_tree"], trees["tgt_tree"]):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["decoder"] + output["encoder"]
        self.val_metric(output["tgt_nll"], batch["tgt_lens"])
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            single_inst = {key: value[:1] for key, value in batch.items()}
            trees = self.forward_visualize(single_inst)
            for src, tgt in zip(trees["src_tree"], trees["tgt_tree"]):
                self.print("Src:", src)
                self.print("Tgt:", tgt)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_best_metric.update(ppl)
        best_ppl = self.val_best_metric.compute()
        self.log("val/ppl", ppl, on_epoch=True, prog_bar=True)
        self.log("val/ppl_best", best_ppl, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        preds = self.forward_inference(batch)["pred"]
        targets = batch["tgt"]

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"preds": preds, "targets": targets, "id": batch["id"]}

    def test_epoch_end(self, outputs) -> None:
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

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = self.hparams.optimizer
        if optimizer_cfg.groups is None or len(optimizer_cfg.groups) == 0:
            params = self.parameters()
        else:
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.named_parameters():
                matches = [
                    i
                    for i, g in enumerate(optimizer_cfg.groups)
                    if re.match(g.pattern, name)
                ]
                if len(matches) > 1:
                    log.warning(
                        f"{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}"
                    )
                if len(matches) > 0:
                    log.debug(
                        f"{name} match {optimizer_cfg.groups[matches[0]].pattern}."
                    )
                    params[matches[0]].append(p)
                else:
                    log.debug(f"{name} match defaults.")
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    log.warning(f"Nothing matches {optimizer_cfg.groups[i].pattern}")
            params = [
                {"params": p, **optimizer_cfg.groups[i]}
                for i, p in enumerate(params)
                if len(p) > 0
            ]
            params.append({"params": default_group})

        optimizer = instantiate(optimizer_cfg.args, params=params, _convert_="all")

        if (scheduler_cfg := self.hparams.scheduler) is None:
            return optimizer

        scheduler_cfg = deepcopy(scheduler_cfg)
        steps_per_epoch = len(self.datamodule.train_dataloader())
        for key in scheduler_cfg.args:
            if isinstance(scheduler_cfg.args[key], str):
                epochs = re.match(r"(\d+) epochs?", scheduler_cfg.args[key])
                if epochs is not None:
                    scheduler_cfg.args[key] = steps_per_epoch * int(epochs.group(1))

        scheduler = instantiate(scheduler_cfg.args, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.interval,
                "frequency": scheduler_cfg.frequency,
                "monitor": scheduler_cfg.monitor,
                "strict": True,
            },
        }


class NeuralPCFGSrcParser(nn.Module):
    def __init__(
        self,
        vocab=100,
        dim=256,
        pt_states=20,
        nt_states=20,
        num_layers=2,
        vocab_out=None,
    ):
        super(NeuralPCFGSrcParser, self).__init__()
        self.vocab = vocab
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.all_states = nt_states + pt_states
        self.pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.root_emb = nn.Parameter(torch.randn(1, dim))
        self.rule_mlp = nn.Sequential(nn.Linear(dim, self.all_states ** 2))
        self.root_mlp = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=nt_states
        )
        if vocab_out is None:
            self.vocab_out = MultiResidualLayer(
                in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab
            )
        else:
            self.vocab_out = vocab_out
        self.neg_huge = -1e5
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.pt_emb)
        nn.init.xavier_normal_(self.nt_emb)
        nn.init.xavier_normal_(self.root_emb)

    def get_rules(self, x: torch.Tensor):
        batch_size, n = x.size()
        root_emb = self.root_emb.expand(batch_size, -1)
        roots = self.root_mlp(root_emb)
        roots = F.log_softmax(roots, 1)
        nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        rules = self.rule_mlp(nt_emb)
        rules = F.log_softmax(rules, 2)
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)
        terms = F.log_softmax(self.vocab_out(pt_emb), 2)
        terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
        x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
        terms = torch.gather(terms, 3, x_expand).squeeze(3)
        return terms, rules, roots

    def forward(self, x, lengths):
        params = self.get_rules(x)
        # for v in params:
        #     assert not v.isnan().any()
        dist = SentCFG(params, lengths)
        return dist

    def marginals(self, x, lengths, argmax=False):
        params = self.get_rules(x)
        dist = SentCFG(params, lengths)
        log_Z = dist.partition
        marginals = dist.marginals[-1]
        return -log_Z, marginals.sum(-1)

    @torch.enable_grad()
    def sample(self, x, lengths, dist=None):
        if dist is None:
            dist = self(x, lengths)
        samples = dist._struct(torch_struct.SampledSemiring).marginals(
            dist.log_potentials, lengths=dist.lengths
        )
        log_Z = dist.partition
        logprobs = dist._struct().score(dist.log_potentials, samples) - log_Z
        return samples, logprobs

    def argmax(self, x, lengths, dist=None):
        if dist is None:
            dist = self(x, lengths)

        spans_onehot = dist.argmax
        log_Z = dist.partition
        logprobs = dist._struct().score(dist.log_potentials, spans_onehot) - log_Z
        return spans_onehot, logprobs


class NeuralQCFGTgtParser(nn.Module):
    def __init__(
        self,
        vocab=100,
        dim=256,
        num_layers=3,
        src_dim=256,
        nt_states=0,
        pt_states=0,
        rule_constraint_type=0,
        use_copy=False,
        nt_span_range=[2, 1000],
        pt_span_range=[1, 1],
        num_samples=10,
        check_ppl=False,
        cpd_args=None,
    ):
        super(NeuralQCFGTgtParser, self).__init__()

        self.use_cpd = cpd_args is not None
        if self.use_cpd:
            self.rank_proj = nn.Parameter(torch.randn(dim, cpd_args.rank))
            self.head_ln = nn.LayerNorm(cpd_args.rank)
            self.left_ln = nn.LayerNorm(cpd_args.rank)
            self.right_ln = nn.LayerNorm(cpd_args.rank)

        self.pcfg = FastestTDPCFG() if self.use_cpd else PCFG()
        # self.pcfg = PCFG()
        self.vocab = vocab
        self.src_dim = src_dim
        self.nt_states = nt_states
        self.pt_states = pt_states
        self.dim = dim

        self.num_samples = num_samples
        self.check_ppl = check_ppl

        self.src_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
        self.register_parameter("src_nt_emb", self.src_nt_emb)
        self.src_nt_node_mlp = MultiResidualLayer(
            in_dim=src_dim, res_dim=dim, num_layers=num_layers
        )
        self.src_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
        self.register_parameter("src_pt_emb", self.src_pt_emb)
        self.src_pt_node_mlp = MultiResidualLayer(
            in_dim=src_dim, res_dim=dim, num_layers=num_layers
        )
        self.rule_mlp_parent = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=None
        )
        self.rule_mlp_left = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=None
        )
        self.rule_mlp_right = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=None
        )
        self.root_mlp_child = nn.Linear(dim, 1, bias=False)
        self.vocab_out = MultiResidualLayer(
            in_dim=dim, res_dim=dim, num_layers=num_layers, out_dim=vocab
        )
        self.neg_huge = -1e5
        self.rule_constraint_type = rule_constraint_type
        self.use_copy = use_copy
        self.nt_span_range = nt_span_range
        self.pt_span_range = pt_span_range

    def decode(self, node_features, spans, tokenizer: Vocabulary):
        # if check_ppl=True, I will compute ppl for samples, return the one with minimum ppl
        # else, just return the one with the maximum score
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans
        )
        preds = self.pcfg.sampled_decoding(
            params,
            self.nt_spans,
            self.nt_states,
            self.pt_spans,
            self.pt_states,
            num_samples=self.num_samples,
            use_copy=self.use_copy,
            max_length=100,
        )

        if self.check_ppl:
            new_preds = []
            for i, preds_inst in enumerate(preds):
                _ids = [
                    inst[0] for inst in preds_inst if len(inst[0]) < 60
                ]  # oom if >=60, ~20gb
                _ids.sort(key=lambda x: len(x), reverse=True)
                _lens = [len(inst) for inst in _ids]
                _ids_t = torch.full((len(_ids), _lens[0]), tokenizer.pad_token_id)
                for j, (snt, length) in enumerate(zip(_ids, _lens)):
                    _ids_t[j, :length] = torch.tensor(snt)
                _ids_t = _ids_t.to(node_features[0].device)
                _node_ft = [node_features[i] for _ in range(len(_ids))]
                _spans = [spans[i] for _ in range(len(_ids))]
                nll = self(_ids_t, _lens, _node_ft, _spans).detach().cpu().numpy()
                ppl = np.exp(nll / np.array(_lens))
                chosen = np.argmin(ppl)
                new_preds.append((preds_inst[chosen][0], -nll))
            preds = new_preds
        else:
            preds = [inst[0] for inst in preds]

        pred_strings = []
        for pred in preds:
            snt, score = pred
            try:
                pred_strings.append(tokenizer.convert_ids_to_tokens(snt))
            except IndexError:
                print("bad pred:", snt)
                pred_strings.append([""])
        return pred_strings

    def get_nt_copy_spans(self, x, span, x_str):
        bsz, N = x.size()
        copy_span = [None for _ in range(N)]
        max_span = max([len(s) for s in span])
        for w in range(1, N):
            c = torch.zeros(bsz, 1, N - w, self.nt_states * max_span).to(x.device)
            mask = torch.zeros_like(c)
            c2 = c[:, :, :, : self.nt_states * max_span].view(
                bsz, 1, N - w, self.nt_states, max_span
            )
            mask2 = mask[:, :, :, : self.nt_states * max_span].view(
                bsz, 1, N - w, self.nt_states, max_span
            )
            c2[:, :, :, -1].fill_(self.neg_huge * 10)
            mask2[:, :, :, -1].fill_(1.0)
            for b in range(bsz):
                l = N
                for i in range(l - w):
                    j = i + w
                    for k, s in enumerate(span[b]):
                        if s[-1] is not None:
                            copy_str = " ".join(s[-1])
                            if " ".join(x_str[b][i : j + 1]) == copy_str:
                                c2[b, :, i, -1, k] = 0
            copy_span[w] = (c, mask)
        return copy_span

    def get_params(self, node_features, spans, x=None, x_str=None):
        # TODO: separate processing terms from get_params. a new func (in=term rules, seq) (out=current output)
        # TODO: copy mechanism
        if self.use_cpd:
            return self.get_params_cpd(node_features, spans, x, x_str)
        else:
            return self.get_params_vanilla(node_features, spans, x, x_str)

    def get_params_cpd(self, node_features, spans, x=None, x_str=None):
        batch_size = len(spans)

        # seperate nt and pt features according to span width
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for span, node_feature in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for i, s in enumerate(span):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(node_feature[i])
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(node_feature[i])
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_feature[-1])
                nt_span.append(span[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_node_features = pad_sequence(
            nt_node_features, batch_first=True, padding_value=0.0
        )
        pt_node_features = pad_sequence(
            pt_node_features, batch_first=True, padding_value=0.0
        )
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
        device = nt_node_features.device
        self.pt_spans = pt_spans
        self.nt_spans = nt_spans

        # e = u + h
        nt_emb = []
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        src_nt_emb = self.src_nt_emb.unsqueeze(0).expand(
            batch_size, self.nt_states, self.dim
        )
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb.append(src_nt_emb)
        nt_emb = torch.cat(nt_emb, 1)
        pt_emb = []
        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        src_pt_emb = self.src_pt_emb.unsqueeze(0).expand(
            batch_size, self.pt_states, self.dim
        )
        src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        src_pt_emb = src_pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
        pt_emb.append(src_pt_emb)
        pt_emb = torch.cat(pt_emb, 1)

        # S->A
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, -1)
        roots += self.neg_huge
        # root must align to root
        for s in range(self.nt_states):
            roots[:, s * nt_num_nodes + nt_num_nodes - 1] -= self.neg_huge
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb) @ self.rank_proj
        rule_emb_left = self.rule_mlp_left(all_emb) @ self.rank_proj
        rule_emb_right = self.rule_mlp_right(all_emb) @ self.rank_proj
        rule_emb_parent = self.head_ln(rule_emb_parent).softmax(-1)
        rule_emb_left = self.left_ln(rule_emb_left).softmax(-2)
        rule_emb_right = self.right_ln(rule_emb_right).softmax(-2)
        # rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        # rule_emb_child = rule_emb_child.view(batch_size, (nt+pt)**2, self.dim)
        # rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1,2))
        # rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        src_pt = pt

        # src_nt_idx = slice(0, src_nt)
        # src_pt_idx = slice(src_nt + tgt_nt, src_nt + tgt_nt + src_pt)
        # tgt_nt_idx = slice(src_nt, src_nt + tgt_nt)
        # tgt_pt_idx = slice(src_nt + tgt_nt + src_pt, src_nt + tgt_nt + src_pt + tgt_pt)

        # if self.rule_constraint_type > 0:
        #   if self.rule_constraint_type == 1:
        #     mask = self.get_rules_mask1(batch_size, nt_num_nodes, pt_num_nodes,
        #                                 nt_spans, pt_spans, device)
        #   elif self.rule_constraint_type == 2:
        #     mask = self.get_rules_mask2(batch_size, nt_num_nodes, pt_num_nodes,
        #                                 nt_spans, pt_spans, device)

        #   rules[:, src_nt_idx, src_nt_idx, src_nt_idx] += mask[:, :, :src_nt, :src_nt]
        #   rules[:, src_nt_idx, src_nt_idx, src_pt_idx] += mask[:, :, :src_nt, src_nt:]
        #   rules[:, src_nt_idx, src_pt_idx, src_nt_idx] += mask[:, :, src_nt:, :src_nt]
        #   rules[:, src_nt_idx, src_pt_idx, src_pt_idx] += mask[:, :, src_nt:, src_nt:]

        # rules = rules
        # rules = rules.view(batch_size, nt, (nt+pt)**2).log_softmax(2).view(
        #   batch_size, nt, nt+pt, nt+pt)

        terms = F.log_softmax(self.vocab_out(pt_emb), 2)

        if x is not None:
            n = x.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if self.use_copy:
                copy_pt = (
                    torch.zeros(batch_size, n, pt).fill_(self.neg_huge * 0.1).to(device)
                )
                copy_pt_view = copy_pt[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                for b in range(batch_size):
                    for c, s in enumerate(pt_spans[b]):
                        if s[-1] == None:
                            continue
                        copy_str = " ".join(s[-1])
                        for j in range(n):
                            if x_str[b][j] == copy_str:
                                copy_pt_view[:, j, -1, c] = 0.0
                copy_mask = torch.zeros_like(copy_pt)
                copy_mask_view = copy_mask[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                # COPY is a special nonterminal
                copy_mask_view[:, :, -1].fill_(1.0)
                # copy_pt has binary weight
                terms = terms * (1 - copy_mask) + copy_pt * copy_mask
                copy_nt = self.get_nt_copy_spans(x, nt_spans, x_str)
            else:
                copy_nt = None
        # TODO: copy_nt
        params = {
            "term": terms,
            "root": roots,
            "left": rule_emb_left,
            "right": rule_emb_right,
            "head": rule_emb_parent,
        }
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def get_params_vanilla(self, node_features, spans, x=None, x_str=None):
        batch_size = len(spans)

        # seperate nt and pt features according to span width
        # TODO sanity check: the root node must be the last element of nt_spans
        pt_node_features, nt_node_features = [], []
        pt_spans, nt_spans = [], []
        for span, node_feature in zip(spans, node_features):
            pt_node_feature = []
            nt_node_feature = []
            pt_span = []
            nt_span = []
            for i, s in enumerate(span):
                s_len = s[1] - s[0] + 1
                if s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]:
                    nt_node_feature.append(node_feature[i])
                    nt_span.append(s)
                if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
                    pt_node_feature.append(node_feature[i])
                    pt_span.append(s)
            if len(nt_node_feature) == 0:
                nt_node_feature.append(node_feature[-1])
                nt_span.append(span[-1])
            pt_node_features.append(torch.stack(pt_node_feature))
            nt_node_features.append(torch.stack(nt_node_feature))
            pt_spans.append(pt_span)
            nt_spans.append(nt_span)
        nt_num_nodes_list = [len(inst) for inst in nt_node_features]
        pt_num_nodes_list = [len(inst) for inst in pt_node_features]
        nt_node_features = pad_sequence(
            nt_node_features, batch_first=True, padding_value=0.0
        )
        pt_node_features = pad_sequence(
            pt_node_features, batch_first=True, padding_value=0.0
        )
        pt_num_nodes = pt_node_features.size(1)
        nt_num_nodes = nt_node_features.size(1)
        device = nt_node_features.device
        self.pt_spans = pt_spans
        self.nt_spans = nt_spans

        # e = u + h
        src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
        src_nt_emb = self.src_nt_emb.unsqueeze(0).expand(
            batch_size, self.nt_states, self.dim
        )
        src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
        src_nt_emb = src_nt_emb.view(batch_size, self.nt_states * nt_num_nodes, -1)
        nt_emb = src_nt_emb

        src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
        src_pt_emb = self.src_pt_emb.unsqueeze(0).expand(
            batch_size, self.pt_states, self.dim
        )
        src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
        src_pt_emb = src_pt_emb.view(batch_size, self.pt_states * pt_num_nodes, -1)
        pt_emb = src_pt_emb

        # S->A
        # TODO we can only use src_root and remove all other src spans here
        roots = self.root_mlp_child(nt_emb)
        roots = roots.view(batch_size, self.nt_states, nt_num_nodes)
        roots += self.neg_huge
        # root must align to root
        for i, n in enumerate(nt_num_nodes_list):
            roots[i, :, n - 1] -= self.neg_huge
        roots = roots.view(batch_size, -1)
        roots = F.log_softmax(roots, 1)

        # A->BC
        nt = nt_emb.size(1)
        pt = pt_emb.size(1)
        all_emb = torch.cat([nt_emb, pt_emb], 1)
        rule_emb_parent = self.rule_mlp_parent(nt_emb)  # b x nt_all x dm
        rule_emb_left = self.rule_mlp_left(all_emb)
        rule_emb_right = self.rule_mlp_right(all_emb)

        rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
        rule_emb_child = rule_emb_child.view(batch_size, (nt + pt) ** 2, self.dim)
        rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1, 2))
        rules = rules.view(batch_size, nt, nt + pt, nt + pt)

        # fmt: off
        nt_mask = torch.arange(nt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(nt_num_nodes_list, device=rules.device).unsqueeze(1)
        pt_mask = torch.arange(pt_num_nodes, device=rules.device).unsqueeze(0) \
            < torch.tensor(pt_num_nodes_list, device=rules.device).unsqueeze(1)
        # fmt: on

        # TODO memory efficient way
        lhs_mask = (
            nt_mask.unsqueeze(1).expand(-1, self.nt_states, -1).reshape(batch_size, -1)
        )
        _pt_rhs_mask = (
            pt_mask.unsqueeze(1).expand(-1, self.pt_states, -1).reshape(batch_size, -1)
        )
        rhs_mask = torch.cat([lhs_mask, _pt_rhs_mask], dim=1)
        mask = torch.einsum("bx,by,bz->bxyz", lhs_mask, rhs_mask, rhs_mask)
        rules[~mask] = self.neg_huge

        src_nt = nt
        src_pt = pt

        src_nt_idx = slice(0, src_nt)
        src_pt_idx = slice(src_nt, src_nt + src_pt)

        if self.rule_constraint_type > 0:
            if self.rule_constraint_type == 1:
                mask = self.get_rules_mask1(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )
            elif self.rule_constraint_type == 2:
                mask = self.get_rules_mask2(
                    batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
                )

            rules[:, src_nt_idx, src_nt_idx, src_nt_idx] += mask[:, :, :src_nt, :src_nt]
            rules[:, src_nt_idx, src_nt_idx, src_pt_idx] += mask[:, :, :src_nt, src_nt:]
            rules[:, src_nt_idx, src_pt_idx, src_nt_idx] += mask[:, :, src_nt:, :src_nt]
            rules[:, src_nt_idx, src_pt_idx, src_pt_idx] += mask[:, :, src_nt:, src_nt:]

        rules = rules
        rules = (
            rules.view(batch_size, nt, (nt + pt) ** 2)
            .log_softmax(2)
            .view(batch_size, nt, nt + pt, nt + pt)
        )

        # A->a
        terms = self.vocab_out(pt_emb)
        terms[~pt_mask] = self.neg_huge
        terms = F.log_softmax(terms, 2)

        if x is not None:
            n = x.size(1)
            terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
            x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
            terms = torch.gather(terms, 3, x_expand).squeeze(3)
            if self.use_copy:
                copy_pt = (
                    torch.zeros(batch_size, n, pt).fill_(self.neg_huge * 0.1).to(device)
                )
                copy_pt_view = copy_pt[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                for b in range(batch_size):
                    for c, s in enumerate(pt_spans[b]):
                        if s[-1] == None:
                            continue
                        copy_str = " ".join(s[-1])
                        for j in range(n):
                            if x_str[b][j] == copy_str:
                                copy_pt_view[:, j, -1, c] = 0.0
                copy_mask = torch.zeros_like(copy_pt)
                copy_mask_view = copy_mask[:, :, :src_pt].view(
                    batch_size, n, self.pt_states, pt_num_nodes
                )
                # COPY is a special nonterminal
                copy_mask_view[:, :, -1].fill_(1.0)
                # copy_pt has binary weight
                terms = terms * (1 - copy_mask) + copy_pt * copy_mask
                copy_nt = self.get_nt_copy_spans(x, nt_spans, x_str)
            else:
                copy_nt = None
            params = (terms, rules, roots, None, None, copy_nt)
        else:
            params = (terms, rules, roots)

        params = {"term": terms, "root": roots, "rule": rules}
        return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

    def get_rules_mask1(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the parent of a j and a k.
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        nt_node_mask = torch.ones(batch_size, nt_num_nodes, nt_num_nodes).to(device)
        pt_node_mask = torch.ones(batch_size, nt_num_nodes, pt_num_nodes).to(device)

        def is_parent(parent, child):
            if child[0] >= parent[0] and child[1] <= parent[1]:
                return True
            else:
                return False

        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            for i, parent_span in enumerate(nt_span):
                for j, child_span in enumerate(nt_span):
                    if not (is_parent(parent_span, child_span)):
                        nt_node_mask[b, i, j].fill_(0.0)
                for j, child_span in enumerate(pt_span):
                    if not (is_parent(parent_span, child_span)):
                        pt_node_mask[b, i, j].fill_(0.0)

        nt_node_mask = (
            nt_node_mask[:, None, :, None, :]
            .expand(
                batch_size, self.nt_states, nt_num_nodes, self.nt_states, nt_num_nodes,
            )
            .contiguous()
        )
        pt_node_mask = (
            pt_node_mask[:, None, :, None, :]
            .expand(
                batch_size, self.nt_states, nt_num_nodes, self.pt_states, pt_num_nodes,
            )
            .contiguous()
        )

        nt_node_mask = nt_node_mask.view(batch_size, nt, nt)
        pt_node_mask = pt_node_mask.view(batch_size, nt, pt)
        node_mask = torch.cat([nt_node_mask, pt_node_mask], 2)
        node_mask = node_mask.unsqueeze(3) * node_mask.unsqueeze(2)
        node_mask = node_mask.view(batch_size, nt, (nt + pt) ** 2)
        node_mask = (1.0 - node_mask) * self.neg_huge
        return node_mask.view(batch_size, nt, nt + pt, nt + pt)

    def get_rules_mask2(
        self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device
    ):
        # A[a i]->B[a j] C[a k], a i must be the DIRECT parent of a j and a k, j!=k.
        #   if a i has no child, a j/k = a i.
        # TODO review this comment
        nt = nt_num_nodes * self.nt_states
        pt = pt_num_nodes * self.pt_states
        bsz = batch_size
        src_nt = self.nt_states
        src_pt = self.pt_states
        node_nt = nt_num_nodes
        node_pt = pt_num_nodes
        node_mask = torch.zeros(
            bsz,
            src_nt * node_nt,
            src_nt * node_nt + src_pt * node_pt,
            src_nt * node_nt + src_pt * node_pt,
        ).to(device)

        nt_idx = slice(0, src_nt * node_nt)
        pt_idx = slice(src_nt * node_nt, src_nt * node_nt + src_pt * node_pt)

        nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_nt, node_nt
        )
        nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_nt, node_nt, src_pt, node_pt
        )
        nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_nt, node_nt
        )
        nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(
            bsz, src_nt, node_nt, src_pt, node_pt, src_pt, node_pt
        )

        def is_parent(parent, child):
            if child[0] >= parent[0] and child[1] <= parent[1]:
                return True
            else:
                return False

        def is_strict_parent(parent, child):
            return is_parent(parent, child) and parent != child

        def span_len(span):
            return span[1] - span[0] + 1

        def covers(parent, child1, child2):
            return (span_len(parent) == (span_len(child1) + span_len(child2))) and (
                (parent[0] == child1[0] and parent[1] == child2[1])
                or (parent[0] == child2[0] and parent[1] == child1[1])
            )

        def overlaps(span1, span2):
            return is_parent(span1, span2) or is_parent(span2, span1)

        # fill_(1.) = not masked
        for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):
            min_nt_span = min([span_len(s) for s in nt_span])
            for i, parent in enumerate(nt_span):
                if span_len(parent) == min_nt_span:
                    nt_ntnt[b, :, i, :, i, :, i].fill_(1.0)
                    for j, child in enumerate(pt_span):
                        if is_strict_parent(parent, child):
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                if span_len(parent) == 1:
                    for j, child in enumerate(pt_span):
                        if parent == child:
                            nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
                            nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
                            nt_ptpt[b, :, i, :, j, :, j].fill_(1.0)
                for j, child1 in enumerate(nt_span):
                    for k, child2 in enumerate(nt_span):
                        if covers(parent, child1, child2):
                            nt_ntnt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ntnt[b, :, i, :, k, :, j].fill_(1.0)
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ntpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptnt[b, :, i, :, k, :, j].fill_(1.0)
                for j, child1 in enumerate(pt_span):
                    for k, child2 in enumerate(pt_span):
                        if covers(parent, child1, child2):
                            nt_ptpt[b, :, i, :, j, :, k].fill_(1.0)
                            nt_ptpt[b, :, i, :, k, :, j].fill_(1.0)

        node_mask = (1.0 - node_mask) * self.neg_huge

        return node_mask.contiguous().view(batch_size, nt, nt + pt, nt + pt)

    def forward(self, x, lengths, node_features, spans, x_str=None, argmax=False):
        params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
            node_features, spans, x, x_str=x_str
        )
        out = self.pcfg(params, lengths, argmax)
        if not argmax:
            return out
        else:
            # out: list of list, containing spans (i, j, label)
            src_nt_states = self.nt_states * nt_num_nodes
            src_pt_states = self.pt_states * pt_num_nodes
            all_spans_node = []
            for b, (all_span, pt_span, nt_span) in enumerate(
                zip(out, pt_spans, nt_spans)
            ):
                all_span_node = []
                for l, r, label in all_span:
                    if l == r:
                        if label < src_pt_states:
                            all_span_node.append(pt_span[label % pt_num_nodes])
                        else:
                            # these are for tgt_nt_states, which are removed for now.
                            all_span_node.append([-1, -1, label - src_pt_states])
                    else:
                        if label < src_nt_states:
                            all_span_node.append(nt_span[label % nt_num_nodes])
                        else:
                            all_span_node.append([-1, -1, label - src_nt_states])
                all_spans_node.append(all_span_node)
            return out, all_spans_node

