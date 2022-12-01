import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from ..components.common import MultiResidualLayer
from .base import ReconstructorBase

logger = logging.getLogger(__file__)


class Simple2Reconstructor(ReconstructorBase):
    def __init__(
        self,
        embedding,
        encoder,
        datamodule,
        kl_scheduler=None,
        dropout=0.2,
        hidden_size=128,
        num_layers=3,
        use_lm_head=False,
    ) -> None:
        super().__init__()

        self.embedding = instantiate(embedding, num_embeddings=len(datamodule.tgt_vocab))
        self.encoder = instantiate(encoder, self.embedding.weight.shape[1])
        self.dropout = nn.Dropout(dropout)
        self.mlp = MultiResidualLayer(
            self.encoder.get_output_dim(), hidden_size, len(datamodule.src_vocab), num_layers=num_layers
        )
        if use_lm_head:
            logger.info("Using LM aux task. Make sure encoder is uni-directional.")
            self.lm_mlp = nn.Linear(self.encoder.get_output_dim(), len(datamodule.tgt_vocab))
            if self.lm_mlp.weight.shape == self.embedding.weight.shape:
                self.lm_mlp.weight = self.embedding.weight
        else:
            self.lm_mlp = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=datamodule.src_vocab.pad_token_id)
        self.kl_scheduler = iter(instantiate(kl_scheduler)) if kl_scheduler is not None else None

    def encode(self, batch):
        seq_ids, seq_lens = batch["tgt_ids"], batch["tgt_lens"]
        x = self.embedding(seq_ids)
        hidden_size = x.shape[-1]
        x = torch.cat(
            [seq_ids.new_zeros(len(seq_ids), 1, hidden_size), x, seq_ids.new_zeros(len(seq_ids), 1, hidden_size)],
            dim=1,
        )
        x[torch.arange(len(seq_ids)), torch.tensor(seq_lens) + 1] = 0.0
        x = self.encoder(x, [item + 2 for item in seq_lens])
        seq_h = x[:, 1:-1]
        x = torch.cat(
            [
                x[:, :-1, : hidden_size // 2],
                x[:, 1:, hidden_size // 2 :],
            ],
            -1,
        )
        span_h = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
        return seq_h, span_h

    def forward(self, batch, tgt_pred):
        tgt_lens = batch["tgt_lens"]
        batch_size = len(tgt_lens)
        seq_h, span_h = self.encode(batch)
        batch_size = len(tgt_lens)
        device = batch["tgt_ids"].device
        loss = 0

        m_term, m_trace = tgt_pred.dist.marginal_with_grad
        m_term = m_term.view(batch_size, -1, tgt_pred.pt_states, tgt_pred.pt_num_nodes).sum(2)
        m_term = m_term.transpose(1, 2)
        linked_pt = F.gumbel_softmax((m_term + 1e-9).log(), tau=1.0, hard=True, dim=2)
        h = torch.matmul(linked_pt, seq_h)

        h = self.dropout(h)
        h = self.mlp(h)

        loss = self.criterion(h.view(-1, h.shape[-1]), batch["src_ids"].flatten())

        q_dist = m_term / (m_term.sum(2, keepdim=True) + 1e-9)
        p_unif = torch.ones(batch_size, max(batch["src_lens"]), max(batch["tgt_lens"]), device=device)
        p_unif /= torch.tensor(tgt_lens, device=device)[:, None, None]
        src_lens_t = torch.tensor(batch["src_lens"], device=device)
        mask = torch.arange(max(batch["src_lens"]), device=device).unsqueeze(0) < src_lens_t.unsqueeze(1)
        kld = mask * (q_dist * (q_dist.clamp(1e-9).log() - p_unif.clamp(1e-9).log())).sum(-1)
        kld = kld.sum(1) / src_lens_t
        # loss is reduced to scalar, this is ok because at last we use the averaged loss in a batch.

        if self.kl_scheduler is not None:
            beta = next(self.kl_scheduler)
        else:
            beta = 1.0

        if self.lm_mlp is not None:
            predicted = self.lm_mlp(seq_h[:, :-1])
            target = batch["tgt_ids"][:, 1:].contiguous().flatten()
            lm_loss = self.criterion(predicted.view(-1, predicted.shape[-1]), target)
        else:
            lm_loss = torch.tensor(0.0)
        return loss + beta * kld, {"vae_beta": torch.tensor(beta), "lm_loss": lm_loss}
