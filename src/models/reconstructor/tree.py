import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence

from ..components.common import MultiResidualLayer
from ..components.rnn_lm import RNNModel, repackage_hidden
from ..components.rnn_seq2seq import Decoder, Seq2SeqTgtOnly
from .base import ReconstructorBase

logger = logging.getLogger(__file__)


class TreeReconstructor(ReconstructorBase):
    # this should contain a KL(q(t_1) || Unif_tree) but omitted.
    # because a source parser entropy regularizor do the exact same thing.
    # to backprop to q(t_1)'s parameter, i use the REINFORCE estimator
    # the value is in 'neg_reward' with a averaged baseline.

    def __init__(
        self,
        lm,
        datamodule,
        kl_scheduler=None,
        hidden_size=64,
        num_layers=1,
        src_node_dim=None,
    ) -> None:
        super().__init__()

        self.lm: RNNModel = instantiate(lm, tok_len=len(datamodule.tgt_vocab) + 2)
        assert self.lm.prm["direction"] == "both"
        self.lm_bos_id = len(datamodule.tgt_vocab)
        self.lm_eos_id = self.lm_bos_id + 1
        self.lm_pad_id = datamodule.tgt_vocab.pad_token_id

        self.gen_eos_id = len(datamodule.src_vocab)
        self.gen_pad_id = datamodule.src_vocab.pad_token_id

        enc_out_dim = self.lm.prm["tok_hid"] * (2 if self.lm.prm["direction"] == "both" else 1)
        self.tok_generator = MultiResidualLayer(
            enc_out_dim,
            hidden_size,
            len(datamodule.src_vocab),
            num_layers,
        )
        self.phrase_mlp = MultiResidualLayer(enc_out_dim, hidden_size, num_layers=num_layers)
        self.phrase_generator = Seq2SeqTgtOnly(Decoder(len(datamodule.src_vocab) + 1, hidden_size, hidden_size, 0.0))

        self.criterion_gen = nn.CrossEntropyLoss(ignore_index=self.gen_pad_id, reduction="none")
        self.criterion_lm = nn.CrossEntropyLoss(ignore_index=self.lm_pad_id)
        self.kl_scheduler = iter(instantiate(kl_scheduler)) if kl_scheduler is not None else None

        self.register_buffer("baseline", torch.zeros(()))
        self.n = 1

    def forward(self, batch, tgt_pred):
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

        m_term, m_trace = tgt_pred.dist.marginal_with_grad
        m_term = m_term.view(batch_size, -1, tgt_pred.pt_states, tgt_pred.pt_num_nodes).sum(2)
        m_term = m_term.transpose(1, 2)
        linked_pt = F.gumbel_softmax((m_term + 1e-9).log(), tau=1.0, hard=True, dim=2)

        m_trace = m_trace.sum(3)
        m_trace = m_trace.movedim(-1, 1).flatten(2)
        linked_nt = F.gumbel_softmax((m_trace + 1e-9).log(), tau=1.0, hard=True, dim=2)

        word_repr = word_repr.view(-1, batch_size, word_repr.shape[-1]).transpose(0, 1)
        fenc_repr = torch.cat(
            [
                word_repr[:, :-1, : word_repr.shape[-1] // 2],
                word_repr[:, 1:, word_repr.shape[-1] // 2 :],
            ],
            -1,
        )
        span_repr = torch.unsqueeze(fenc_repr, 1) - torch.unsqueeze(fenc_repr, 2)

        h = torch.matmul(linked_pt, word_repr[:, 1:-1])
        output_src = self.tok_generator(h)
        loss_recon = self.criterion_gen(output_src.view(-1, output_src.shape[-1]), src_ids.flatten())
        loss_recon = loss_recon.view(batch_size, -1).mean(-1)

        h = torch.matmul(linked_nt, span_repr.view(batch_size, -1, span_repr.shape[-1]))
        h = h.view(1, -1, h.shape[-1])
        h = self.phrase_mlp(h)

        tgt = []
        for bidx, spans in enumerate(tgt_pred.nt_nodes):
            for span in spans:
                tgt.append(src_ids[bidx, span[0] : span[1]])
            for _ in range(len(spans), tgt_pred.nt_num_nodes):
                tgt.append(torch.tensor([self.gen_pad_id], device=device))
        _lens = [len(item) for item in tgt]
        tgt = pad_sequence(tgt, padding_value=self.gen_pad_id)
        tgt = torch.cat([tgt, torch.full((1, *tgt.shape[1:]), self.gen_pad_id, device=device)], dim=0)
        tgt[_lens, torch.arange(len(_lens))] = self.gen_eos_id
        output_src = self.phrase_generator(h, tgt)
        loss_recon_phrase = self.criterion_gen(output_src.view(-1, output_src.shape[-1]), tgt.flatten())
        loss_recon_phrase = loss_recon_phrase.view(-1, batch_size, tgt_pred.nt_num_nodes).mean((0, 2))

        src_lens_t = torch.tensor(batch["src_lens"], device=device, dtype=torch.float)
        tgt_lens_t = torch.tensor(batch["tgt_lens"], device=device, dtype=torch.float)

        q_dist = m_term / (m_term.sum(2, keepdim=True) + 1e-9)
        p_unif = torch.ones(batch_size, max(batch["src_lens"]), max(batch["tgt_lens"]), device=device)
        p_unif /= torch.tensor(tgt_lens, device=device)[:, None, None]
        kld = (q_dist * (q_dist.clamp(1e-9).log() - p_unif.log())).sum(-1)
        kld = kld.sum(1) / src_lens_t

        q_dist = m_trace / (m_trace.sum(2, keepdim=True) + 1e-9)
        p_unif = torch.ones(batch_size, tgt_pred.nt_num_nodes, (max(batch["tgt_lens"]) + 1) ** 2, device=device)
        p_unif /= (tgt_lens_t * 2 - 3)[:, None, None]
        kld_phrase = (q_dist * (q_dist.clamp(1e-9).log() - p_unif.log())).sum(-1)
        kld_phrase = kld_phrase.sum(1) / src_lens_t

        if self.kl_scheduler is not None:
            beta = next(self.kl_scheduler)
        else:
            beta = 1.0

        with torch.no_grad():
            _neg_reward = loss_recon + loss_recon_phrase
            neg_reward = _neg_reward - self.baseline
            self.baseline = self.baseline * ((self.n - 1) / self.n) + _neg_reward.mean() / self.n
            self.n += 1

        return loss_lm + loss_recon + loss_recon_phrase + beta * (kld + kld_phrase), {
            "vae_beta": torch.tensor(beta),
            "lm_loss": loss_lm,
            "kld": kld + kld_phrase,
            "recon": loss_recon + loss_recon_phrase,
            "rec_neg_reward": neg_reward,
        }

    def forward_lang_only(self, batch, tgt_pred):
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
        lm_ids[torch.arange(batch_size), tgt_lens] = self.lm_eos_id
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
