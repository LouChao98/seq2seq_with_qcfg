import logging
import math
import random
import re
from collections import Counter, defaultdict
from email.policy import default
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from amrlib.utils.logging import silence_penman
from nltk.tokenize import word_tokenize
from numba import jit
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src import is_under_debugger
from src.datamodules.components.vocab import Vocabulary, VocabularyPair
from src.datamodules.datamodule import _DataModule
from vendor.amrlib.amrlib.models.parse_xfm.penman_serializer import load_and_serialize

from .sampler import BucketedSampler, kmeans

logger = logging.getLogger(__file__)


class PenmanDataModule(_DataModule):
    def __init__(
        self,
        train_file,
        dev_file,
        test_file,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
        copy_mode: str = "none",
        transformer_tokenizer_name: str = None,
        batch_size: int = 64,
        token_size: int = 0,  # enable if nonzero
        eval_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        assert copy_mode in ("none", "token", "phrase")
        silence_penman()
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        with self.trace_persistent_variables():
            self.src_vocab: Optional[Vocabulary] = None
            self.tgt_vocab: Optional[Vocabulary] = None
            self.vocab_pair: Optional[VocabularyPair] = None
            self.use_transformer_tokenizer = transformer_tokenizer_name is not None
            if transformer_tokenizer_name is not None:
                self.transformer_tokenizer: PreTrainedTokenizer = (
                    AutoTokenizer.from_pretrained(transformer_tokenizer_name)
                )

            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_train = self.read_file(self.hparams.train_file)
        data_val = self.read_file(self.hparams.dev_file)
        data_test = self.read_file(self.hparams.test_file)

        data_train = [
            inst
            for inst in data_train
            if len(inst["src"]) <= self.hparams.max_src_len
            and len(inst["tgt"]) <= self.hparams.max_tgt_len
        ]
        data_val = [
            inst
            for inst in data_val
            if len(inst["src"]) <= self.hparams.max_src_len
            and len(inst["tgt"]) <= self.hparams.max_tgt_len
        ]

        data_train = self.process_all_copy(data_train)
        data_val = self.process_all_copy(data_val)
        data_test = self.process_all_copy(data_test)

        self.src_vocab, self.tgt_vocab = self.build_vocab(data_train)
        self.vocab_pair = VocabularyPair(self.src_vocab, self.tgt_vocab)

        self.data_train = self.apply_vocab(data_train)
        self.data_val = self.apply_vocab(data_val)
        self.data_test = self.apply_vocab(data_test)

        if self.use_transformer_tokenizer:
            self.data_train = self.apply_transformer_tokenizer(self.data_train)
            self.data_val = self.apply_transformer_tokenizer(self.data_val)
            self.data_test = self.apply_transformer_tokenizer(self.data_test)

    def read_file(self, fpath):
        data = load_and_serialize(fpath, False)
        reentry_pattern = re.compile(r"(.*)_(\d+)")
        converted = []
        for di, (graph, serial, sent) in enumerate(
            zip(data["graphs"], data["serials"], data["sents"])
        ):
            processed_serial = []
            spans = []
            bracket_stack = []
            variables = defaultdict(lambda: defaultdict(list))
            tokens = serial.split()
            for i, c in enumerate(tokens):
                if c == "(":
                    bracket_stack.append(len(processed_serial))
                elif c == ")":
                    left_bracket = bracket_stack.pop()
                    if len(processed_serial) > left_bracket + 1:
                        spans.append((left_bracket, len(processed_serial)))
                    if left_bracket > 0:
                        spans.append((left_bracket - 1, len(processed_serial)))
                    if (
                        len(processed_serial) > 2
                        and tokens[i - 2][0] == ":"
                        and tokens[i - 1] != "("
                    ):
                        # detect constant list such as op lists
                        spans.append((len(processed_serial) - 2, len(processed_serial)))
                elif c[0] == ":":  # rel
                    if (
                        len(processed_serial) > 2
                        and tokens[i - 2][0] == ":"
                        and tokens[i - 1] != "("
                    ):
                        # detect constant list such as op lists
                        spans.append((len(processed_serial) - 2, len(processed_serial)))
                    processed_serial.append(c)
                else:  # variable or constant
                    match = re.match(reentry_pattern, c)
                    if match is None:
                        variables[c][None].append(len(processed_serial))
                        processed_serial.append(c)
                    else:
                        surface, id_ = match.groups()
                        variables[surface][id_].append(len(processed_serial))
                        processed_serial.append(surface)
            assert len(bracket_stack) == 0
            tgt_len = len(processed_serial)
            src = word_tokenize(sent)
            if tgt_len <= 1:  # TODO !!!
                continue
            if len(src) <= 1:
                continue
            inst = {
                "id": di,
                "src": src,
                "tgt": processed_serial,
                "tgt_spans": spans,
                "tgt_mask": self.gen_impossible_span_mask(spans, tgt_len),
                "var": variables,
                "raw": graph,
            }
            converted.append(inst)
        return converted

    def gen_impossible_span_mask(self, spans, length):
        # spans: inclusive start, exclusive end
        # True = impossible
        span_mask = np.zeros((length + 1, length + 1), dtype=np.bool8)
        for left, right in spans:
            span_mask[:left, left + 1 : right] = True
            span_mask[left + 1 : right, right + 1 :] = True
        masks = []
        for w in range(1, length):
            masks.append(
                torch.tensor([span_mask[i, i + w + 1] for i in range(length - w)])
            )
        return masks

    def process_all_copy(self, data):
        # none = do nothing
        # token = token
        # phrase = token + phrase
        if self.hparams.copy_mode == "none":
            return data
        for item in data:
            self.process_token_copy(item)
        if self.hparams.copy_mode == "phrase":
            for item in data:
                self.process_phrase_copy(item)
        return data

    def process_token_copy(self, inst):
        src = inst["src"]  # do not use ids, because src/tgt have diff vocab
        tgt = inst["tgt"]
        copy_m = np.zeros((len(src), len(tgt)), dtype=np.bool8)
        for i, stoken in enumerate(src):
            for j, ttoken in enumerate(tgt):
                if stoken == ttoken:
                    copy_m[i, j] = True
        inst["copy_token"] = copy_m

    def process_phrase_copy(self, inst):
        # generate a list of bool vectors from width 2 to len(tgt)-1.
        src = inst["src"]
        tgt = inst["tgt"]
        output = []

        # width = 2
        copy_position = []
        for i in range(len(src) - 1):
            for j in range(len(tgt) - 1):
                if src[i : i + 2] == tgt[j : j + 2]:
                    copy_position.append((i, j))
        previous = copy_position
        output.append(copy_position)

        # width > 2
        for w in range(3, len(tgt) + 2):
            copy_position = []
            for i, j in previous:
                if i > len(src) - w or j > len(tgt) - w:
                    continue
                if src[i + w - 1] == tgt[j + w - 1]:
                    copy_position.append((i, j))
            previous = copy_position
            output.append(copy_position)
        inst["copy_phrase"] = output

    def build_vocab(self, data):
        src_vocab_cnt = Counter()
        tgt_vocab_cnt = Counter()
        for inst in data:
            src_vocab_cnt.update(inst["src"])
            tgt_vocab_cnt.update(inst["tgt"])
        if self.hparams.copy_mode != "none":
            logger.warning("I set src tokens in tgt vocab due to copy mode.")
            for inst in data:
                tgt_vocab_cnt.update(inst["src"])
        src_vocab = Vocabulary(src_vocab_cnt)
        tgt_vocab = Vocabulary(tgt_vocab_cnt)
        return src_vocab, tgt_vocab

    def apply_vocab(self, data):
        processed = []
        for inst in data:
            processed.append(
                {
                    **inst,
                    "src_ids": self.src_vocab.convert_tokens_to_ids(inst["src"]),
                    "tgt_ids": self.tgt_vocab.convert_tokens_to_ids(inst["tgt"]),
                }
            )
        return processed

    def train_dataloader(self):
        if self.hparams.token_size <= 0:
            loader = DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collator,
                shuffle=False if is_under_debugger() else True,
            )
        else:
            buckets = dict(
                zip(
                    *kmeans(
                        [len(item["tgt"]) for item in self.data_train],
                        math.ceil(math.log2(len(self.data_train))),
                    )
                )
            )
            loader = DataLoader(
                dataset=self.data_train,
                batch_sampler=BucketedSampler(
                    buckets,
                    self.hparams.token_size,
                    True,
                ),
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collator,
            )

        logger.info(f"Train dataloader: {len(loader)}")
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        logger.info(f"Val dataloader: {len(loader)}")
        return loader

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.eval_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )

    def collator(self, data):
        batch_size = len(data)
        tgt_lens = [len(inst["tgt_ids"]) for inst in data]
        argsort = list(range(len(data)))
        argsort.sort(key=lambda i: tgt_lens[i], reverse=True)
        data = [data[i] for i in argsort]

        src = [inst["src"] for inst in data]
        tgt = [inst["tgt"] for inst in data]
        raw = [inst["raw"] for inst in data]
        tgt_spans = [inst["tgt_spans"] for inst in data]
        src_lens = [len(inst["src_ids"]) for inst in data]
        tgt_lens = [len(inst["tgt_ids"]) for inst in data]
        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)
        batched_src_ids = torch.full(
            (len(tgt_lens), max_src_len), self.src_vocab.pad_token_id
        )
        batched_tgt_ids = torch.full(
            (len(tgt_lens), max_tgt_len), self.tgt_vocab.pad_token_id
        )
        for i, inst in enumerate(data):
            s, t = inst["src_ids"], inst["tgt_ids"]
            batched_src_ids[i, : len(s)] = torch.tensor(s)
            batched_tgt_ids[i, : len(t)] = torch.tensor(t)

        tgt_masks = []
        max_tgt_len = max(tgt_lens)
        for wi, w in enumerate(range(1, max_tgt_len)):
            mask = torch.zeros(batch_size, max_tgt_len - w, dtype=torch.bool)
            for bidx, inst in enumerate(data):
                m = inst["tgt_mask"]
                if wi >= len(m):
                    continue
                m = m[wi]
                mask[bidx, : len(m)] = m
            tgt_masks.append(mask)

        # mark positions that should have different alignment.
        # This is only for pt because we use it too group reentrancies.
        # TODO there is a problem: {'man': {"01": [1, 2], "02": [3, 4]}}
        #   We need to constraint (1,2) and (3,4) have different alignments
        #   but same alignments between 1 and 2, and 3 and 4.
        #   Current strategy is only constraining arbitrary one in each group to neq,
        #   e.g., constraint 1 and 3, but lefting (1 to 4) (2 to 3) (2 to 4)
        #   unconstrained.This should be ok is we also constrain (1 and 2) (3 and 4)
        #   have the same alignment
        pt_neq_constraints = []
        for inst, tgt_len in zip(data, tgt_lens):
            pt_neq_constraint = torch.zeros(tgt_len, dtype=torch.float32)
            for groups in inst["var"].values():
                if len(groups) == 1:
                    continue
                for vs in groups.values():
                    # pt_neq_constraint[vs[0]] = 1.0
                    pt_neq_constraint[random.choice(vs)] = 1.0
            pt_neq_constraints.append(pt_neq_constraint)
        pt_neq_constraints = pad_sequence(pt_neq_constraints, batch_first=True)

        # Encourage vars in the same group have the same alignment
        # NOTE only consider at most 5 groups each step
        pt_eq_constraints = []
        for inst, tgt_len in zip(data, tgt_lens):
            constraints_for_groups = [
                l for v in inst["var"].values() for l in v.values() if len(l) > 1
            ]
            constraints_for_groups = random.sample(
                constraints_for_groups, min(3, len(constraints_for_groups))
            )
            pt_eq_constraints.append(
                [torch.tensor(item) for item in constraints_for_groups]
            )

        batched = {}
        batched["id"] = torch.tensor([inst["id"] for inst in data])
        batched["src"] = src
        batched["tgt"] = tgt
        batched["tgt_spans"] = tgt_spans
        batched["tgt_masks"] = tgt_masks
        batched["tgt_pt_neq_constraint"] = pt_neq_constraints
        batched["tgt_pt_eq_constraint"] = pt_eq_constraints
        batched["raw"] = raw
        batched["src_ids"] = batched_src_ids
        batched["tgt_ids"] = batched_tgt_ids
        batched["src_lens"] = src_lens
        batched["tgt_lens"] = tgt_lens

        copy_token = None
        if "copy_token" in data[0]:
            copy_token = torch.zeros(
                len(src), max(src_lens), max(tgt_lens), dtype=torch.bool
            )
            for i, inst in enumerate(data):
                inst = torch.from_numpy(inst["copy_token"])
                copy_token[i, : inst.shape[0], : inst.shape[1]] = inst
            batched["copy_token"] = copy_token

        copy_phrase = None
        if "copy_phrase" in data[0]:
            copy_phrase = [item["copy_phrase"] for item in data]
            batched["copy_phrase"] = copy_phrase

        if self.use_transformer_tokenizer:
            transformer_inp = self.transformer_tokenizer(
                src,
                padding=True,
                is_split_into_words=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping_raw = transformer_inp.pop("offset_mapping")
            offset_mapping = torch.zeros(
                transformer_inp["input_ids"].shape, dtype=np.int64
            )
            for i, mapping in enumerate(offset_mapping_raw):
                cursor = 0
                for j, item in enumerate(mapping):
                    if item[0] == item[1] == 0:
                        if j != 0:
                            break
                        cursor = 0
                    elif item[0] == 0:
                        cursor += 1
                    offset_mapping[i, j] = cursor
            batched["transformer_inputs"] = transformer_inp
            batched["transformer_offset"] = offset_mapping

        if "src_tree" in data[0]:
            trees = [item["src_tree"] for item in data]
            batched["src_tree"] = trees

        return batched


if __name__ == "__main__":

    datamodule = PenmanDataModule(
        "data/AMR/tdata_xfm/dev.txt.nowiki",
        "data/AMR/tdata_xfm/dev.txt.nowiki",
        "data/AMR/tdata_xfm/dev.txt.nowiki",
        copy_mode="phrase",
        batch_size=2,
        enable_cache=False,
    )
    datamodule.setup()
    print("Loaded.")

    print("=" * 80)
    for batch in datamodule.train_dataloader():
        print(batch)
        break
