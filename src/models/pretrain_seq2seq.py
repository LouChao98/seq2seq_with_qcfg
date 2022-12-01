import logging
import operator
from typing import Any, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import MinMetric

from src.models.base import ModelBase
from src.utils.fn import apply_to_nested_tensor, report_ids_when_err
from src.utils.metric import PerplexityMetric

from .components.common import MultiResidualLayer
from .components.rnn_lm import repackage_hidden
from .components.rnn_seq2seq import Decoder, Seq2SeqTgtOnly

log = logging.getLogger(__file__)


class PretrainSeq2SeqModule(ModelBase):
    def __init__(
        self,
        lm=None,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        optimizer=None,
        scheduler=None,
        load_from_checkpoint=None,
        param_initializer="xavier_uniform",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None, datamodule=None) -> None:
        super().setup(stage)

        assert datamodule is not None or self.trainer.datamodule is not None
        self.datamodule = datamodule or self.trainer.datamodule

        self.lm = instantiate(self.hparams.lm, tok_len=len(self.datamodule.tgt_vocab) + 2)
        self.lm_bos_id = len(self.datamodule.tgt_vocab)
        self.lm_eos_id = self.lm_bos_id + 1
        self.lm_pad_id = self.datamodule.tgt_vocab.pad_token_id

        self.gen_eos_id = len(self.datamodule.src_vocab)
        self.gen_pad_id = self.datamodule.src_vocab.pad_token_id

        hidden_size = self.hparams.hidden_size
        num_layers = self.hparams.num_layers
        enc_out_dim = self.lm.prm["tok_hid"] * (2 if self.lm.prm["direction"] == "both" else 1)
        self.phrase_mlp = MultiResidualLayer(enc_out_dim, hidden_size, num_layers=num_layers)
        self.phrase_generator = Seq2SeqTgtOnly(
            Decoder(len(self.datamodule.src_vocab) + 1, hidden_size, hidden_size, self.hparams.dropout)
        )

        self.criterion_gen = nn.CrossEntropyLoss(ignore_index=self.gen_pad_id)
        self.criterion_lm = nn.CrossEntropyLoss(ignore_index=self.lm_pad_id)

        self.train_metric = PerplexityMetric()
        self.ppl_metric = PerplexityMetric()
        self.ppl_best_metric = MinMetric()

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
        src_ids = batch["src_ids"]
        tgt_ids = batch["tgt_ids"]
        tgt_lens = batch["tgt_lens"]
        batch_size = len(tgt_lens)
        device = tgt_ids.device

        lm_ids = torch.cat(
            [
                torch.full((batch_size, 1), self.lm_bos_id, device=tgt_ids.device),
                tgt_ids,
                torch.full((batch_size, 1), self.lm_pad_id, device=tgt_ids.device),
            ],
            dim=1,
        )
        lm_ids[torch.arange(batch_size), torch.tensor(tgt_lens) + 1] = self.lm_eos_id
        lm_batch = {"word": {"index": lm_ids}, "seq_len": max(tgt_lens) + 2}
        hidden = self.lm.init_hidden(lm_batch)
        # Cut the computation graph (Initialize)
        hidden = repackage_hidden(hidden)
        # LongTensor of token_ids [seq_len, batch_size]
        input = self.lm.batch2input(lm_batch, device)
        # target_flat: LongTensor of token_ids [seq_len*batch_size]
        target_flat = self.lm.batch2flat(lm_batch, device)

        # output: [seq_len, nbatch, ntoken], hidden: [nlayer, nbatch, nhid]
        output, word_repr, last_hidden = self.lm(input, hidden)
        # output_flat: LongTensor of token_ids [seq_len*batch_size, ntoken]
        output_flat = output.view(-1, output.shape[2])
        # Calculate the mean of all losses.
        # loss: float
        loss_lm = self.criterion_lm(output_flat, target_flat)

        word_repr = word_repr.view(-1, batch_size, word_repr.shape[-1]).transpose(0, 1)
        fenc_repr = torch.cat(
            [
                word_repr[:, :-1, : word_repr.shape[-1] // 2],
                word_repr[:, 1:, word_repr.shape[-1] // 2 :],
            ],
            -1,
        )
        start_repr = fenc_repr[:, 0]
        end_repr = fenc_repr[torch.arange(batch_size), tgt_lens]
        h = end_repr - start_repr

        h = h.view(1, -1, h.shape[-1])
        h = self.phrase_mlp(h)

        src_ids = src_ids.transpose(0, 1)
        output_src = self.phrase_generator(h, src_ids, 0.5 if self.training else 0.0)
        loss_seq2seq = self.criterion_gen(output_src.view(-1, output_src.shape[-1]), src_ids.flatten())

        logprob = output_src.log_softmax(-1)
        logprob = logprob.gather(2, src_ids.unsqueeze(-1))
        mask = src_ids == self.gen_pad_id
        logprob[mask] = 0.0
        logprob = logprob.squeeze(-1).sum(0)

        return {"loss_lm": loss_lm, "loss_recon": loss_seq2seq, "nll": -logprob}

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        loss = output["loss_lm"].mean() + output["loss_recon"].mean()
        ppl = self.train_metric(output["nll"], batch[f"src_lens"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ppl", ppl, prog_bar=True)
        self.log("train/lm", output["loss_lm"].mean(), prog_bar=True)
        self.log("train/loss_recon", output["loss_recon"].mean(), prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.ppl_metric.reset()

    @torch.inference_mode(False)
    def validation_step(self, batch: Any, batch_idx: int):
        batch = apply_to_nested_tensor(batch, func=lambda x: x.clone())
        output = self(batch)
        self.ppl_metric(output["nll"], batch[f"src_lens"])
        return {"id": batch["id"]}

    def validation_epoch_end(self, outputs: List[Any]):
        ppl = self.ppl_metric.compute()
        self.ppl_best_metric.update(ppl)
        best_ppl = self.ppl_best_metric.compute().item()
        metric = {"ppl": ppl, "best_ppl": best_ppl}
        self.log_dict({"val/" + k: v for k, v in metric.items()})
        self.print(metric)

    @torch.inference_mode(False)
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.ppl_metric.reset()

    @torch.inference_mode(False)
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        ppl = self.ppl_metric.compute()
        metric = {"ppl": ppl}
        self.log_dict({"test/" + k: v for k, v in metric.items()})
        self.print(metric)
