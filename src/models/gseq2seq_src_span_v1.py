import logging
from typing import Any, List, Optional

import numpy as np
import torch

from src.models.general_seq2seq import GeneralSeq2SeqModule
from src.utils.fn import extract_parses

log = logging.getLogger(__file__)


class GSeq2seqSrcSpanV1(GeneralSeq2SeqModule):
    """Based on GeneralSeq2Seq
    + use copy_phrases to bonus src parser. use svm hinge loss
    """

    def __init__(self, *args, hinge_loss_alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

    def forward(self, batch):
        src_ids, src_lens = batch["src_ids"], batch["src_lens"]
        copy_position = (batch.get("copy_token"), batch.get("copy_phrase"))

        dist = self.parser(src_ids, src_lens)
        src_nll = -dist.partition
        src_spans, src_logprob = self.parser.sample(src_ids, src_lens, dist)
        src_spans = extract_parses(src_spans[-1], src_lens, inc=1)[0]

        x = self.encode(batch)
        node_features, node_spans = self.tree_encoder(x, src_lens, spans=src_spans)

        tgt_nll = self.decoder(
            batch["tgt_ids"],
            batch["tgt_lens"],
            node_features,
            node_spans,
            copy_position=copy_position,
        )

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
            copy_position=copy_position,
        )
        neg_reward = (tgt_nll - tgt_nll_argmax).detach()

        copy_phrase = batch["copy_phrase"]
        batch_size, n, *_ = x.shape
        add_scores = [
            np.full(
                (batch_size, n - w),
                0,
                dtype=np.float32,
            )
            for w in range(1, n)
        ]
        for batch_idx, possible_copy in enumerate(copy_phrase):
            for w, maps in enumerate(possible_copy):
                for one_map in maps:
                    add_scores[w][batch_idx, one_map[0]] = 1
        add_scores_ = []
        for item in add_scores:
            add_scores_.append(torch.from_numpy(item).to(x.device).unsqueeze(-1))
        src_spans_argmax_aug, _ = self.parser.argmax(
            src_ids, src_lens, extra_scores={"add": add_scores_}
        )
        auged = dist._struct().score(dist.log_potentials, src_spans_argmax_aug)
        hinge_loss = self.hparams.hinge_loss_alpha * (
            src_logprob_argmax - auged + 1
        ).clamp(0)

        return {
            "decoder": tgt_nll.mean(),
            "encoder": src_nll.mean()
            + (src_logprob * neg_reward).mean()
            + hinge_loss.mean(),
            "tgt_nll": tgt_nll.mean(),
            "src_nll": src_nll.mean(),
            "reward": -neg_reward.mean(),
        }
