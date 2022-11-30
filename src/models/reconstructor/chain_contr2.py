import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from supar.modules.affine import Biaffine

from ..components.common import MultiResidualLayer
from ..components.rnn_lm import RNNModel, repackage_hidden
from .base import ReconstructorBase

logger = logging.getLogger(__file__)

# ContrastiveReconstructor aims to choose the correct tgt token instead of generating it.


class ChainContrastive2Reconstructor(ReconstructorBase):
    def __init__(
        self,
        lm,
        datamodule,
        src_node_dim,
        hidden_dim,
        num_layers=1,
        kl_scheduler=None,
    ) -> None:
        super().__init__()

        self.lm: RNNModel = instantiate(lm, tok_len=len(datamodule.tgt_vocab) + 2)
        self.bos_id = len(datamodule.tgt_vocab)
        self.eos_id = self.bos_id + 1
        self.pad_id = datamodule.tgt_vocab.pad_token_id

        self.src_mlp = MultiResidualLayer(src_node_dim, hidden_dim, num_layers=num_layers)
        self.tgt_mlp = MultiResidualLayer(
            self.lm.prm["tok_hid"] * (2 if self.lm.prm["direction"] == "both" else 1),
            hidden_dim,
            num_layers=num_layers,
        )
        self.scorer = Biaffine(hidden_dim, 1)

        self.criterion_lm = nn.CrossEntropyLoss(ignore_index=datamodule.tgt_vocab.pad_token_id)
        self.kl_scheduler = iter(instantiate(kl_scheduler)) if kl_scheduler is not None else None

    def forward(self, batch, tgt_pred):
        tgt_ids = batch["tgt_ids"]
        tgt_lens = batch["tgt_lens"]
        batch_size = len(tgt_lens)
        device = tgt_ids.device

        lm_ids = torch.cat(
            [
                torch.full((batch_size, 1), self.bos_id, device=tgt_ids.device),
                tgt_ids,
                torch.full((batch_size, 1), self.pad_id, device=tgt_ids.device),
            ],
            dim=1,
        )
        lm_ids[torch.arange(batch_size), torch.tensor(tgt_lens) + 1] = self.eos_id
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

        m_term, m_trace = tgt_pred.dist.marginal_with_grad
        m_term = m_term.view(batch_size, -1, tgt_pred.pt_states, tgt_pred.pt_num_nodes).sum(2)
        m_term = m_term.transpose(1, 2)
        q_dist = m_term / (m_term.sum(2, keepdim=True) + 1e-9)

        word_repr = word_repr.view(-1, batch_size, word_repr.shape[-1])
        if self.lm.prm["direction"] == "both":
            word_repr = word_repr[1:-1]
        elif self.lm.prm["direction"] == "left2right":
            word_repr = word_repr[1:]
        else:
            word_repr = word_repr[:-1]

        tgt_words = self.tgt_mlp(word_repr.transpose(0, 1))
        src_words = self.src_mlp(tgt_pred.pt_features)
        score = (
            self.scorer(tgt_words, src_words).unsqueeze(1).expand(-1, tgt_pred.pt_num_nodes, -1, -1)
        )  # b x (so) x to x so

        src_lens_t = torch.tensor(batch["src_lens"], device=device)
        mask = torch.arange(max(batch["src_lens"]), device=device).unsqueeze(0) < src_lens_t.unsqueeze(1)

        loss_recon = (score.max(3).values - score.diagonal(0, dim1=1, dim2=3).transpose(1, 2) + 1).clamp(0)
        loss_recon = (loss_recon * q_dist).sum(2)
        loss_recon = (loss_recon * mask).sum(1)

        p_unif = torch.ones(batch_size, max(batch["src_lens"]), max(batch["tgt_lens"]), device=device)
        p_unif /= torch.tensor(tgt_lens, device=device)[:, None, None]
        kld = mask * (q_dist * (q_dist.clamp(1e-9).log() - p_unif.log())).sum(-1)
        kld = kld.sum(1) / src_lens_t
        # loss is reduced to scalar, this is ok because at last we use the averaged loss in a batch.

        if self.kl_scheduler is not None:
            beta = next(self.kl_scheduler)
        else:
            beta = 1.0

        return loss_lm + loss_recon + beta * kld, {
            "vae_beta": torch.tensor(beta),
            "lm_loss": loss_lm,
            "kld": kld,
            "recon": loss_recon,
        }

    def forward_lang_only(self, batch, tgt_pred):
        tgt_ids = batch["tgt_ids"]
        tgt_lens = batch["tgt_lens"]
        batch_size = len(tgt_lens)
        device = tgt_ids.device

        lm_ids = torch.cat(
            [
                torch.full((batch_size, 1), self.bos_id, device=tgt_ids.device),
                tgt_ids,
                torch.full((batch_size, 1), self.pad_id, device=tgt_ids.device),
            ],
            dim=1,
        )
        lm_ids[torch.arange(batch_size), tgt_lens] = self.eos_id
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

        return loss_lm, {"lm_loss": loss_lm}
