import json
import logging
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from email.policy import default
from pathlib import Path
from tokenize import single_quoted
from typing import Any, List, Optional, Union

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src import is_under_debugger
from src.datamodules.components.vocab import Vocabulary, VocabularyPair
from src.datamodules.datamodule import _DataModule

from .sampler import BucketedSampler, kmeans

logger = logging.getLogger(__file__)


class SemanticParsingDataModule(_DataModule):
    def __init__(
        self,
        train_file,
        dev_file,
        test_file,
        prior_alignment_file: str = None,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
        copy_mode: str = "none",
        add_eos: bool = False,
        transformer_tokenizer_name: str = None,
        batch_size: int = 64,
        token_size: int = 0,  # enable if nonzero
        eval_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        assert copy_mode in ("none", "token", "phrase")

        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        with self.trace_persistent_variables():
            self.src_vocab: Optional[Vocabulary] = None
            self.tgt_vocab: Optional[Vocabulary] = None
            self.vocab_pair: Optional[VocabularyPair] = None
            self.use_transformer_tokenizer = transformer_tokenizer_name is not None
            if transformer_tokenizer_name is not None:
                extra_args = {}
                if transformer_tokenizer_name.startswith("roberta"):
                    extra_args["add_prefix_space"] = True
                self.transformer_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    transformer_tokenizer_name, **extra_args
                )

            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_train = self.read_file(self.hparams.train_file)
        data_val = self.read_file(self.hparams.dev_file)
        data_test = self.read_file(self.hparams.test_file)
        if self.hparams.prior_alignment_file:
            data_train = self.load_prior_alignment(data_train, self.hparams.prior_alignment_file)

        _num_orig_train = len(data_train)
        data_train = [
            inst
            for inst in data_train
            if len(inst["src"]) <= self.hparams.max_src_len and len(inst["tgt"]) <= self.hparams.max_tgt_len
        ]
        if (d := _num_orig_train - len(data_train)) > 0:
            logger.warning(f"Dropping {d} samples in TrainingSet.")
        _num_orig_Val = len(data_val)
        data_val = [
            inst
            for inst in data_val
            if len(inst["src"]) <= self.hparams.max_src_len and len(inst["tgt"]) <= self.hparams.max_tgt_len
        ]
        if (d := _num_orig_Val - len(data_val)) > 0:
            logger.warning(f"Dropping {d} samples in ValSet.")

        data_train = self.process_all_copy(data_train)
        data_val = self.process_all_copy(data_val)
        data_test = self.process_all_copy(data_test)

        if self.hparams.add_eos:
            data_train = self.add_eos(data_train)
            data_val = self.add_eos(data_val)
            data_test = self.add_eos(data_test)

        self.src_vocab, self.tgt_vocab = self.build_vocab(data_train)
        self.vocab_pair = VocabularyPair(self.src_vocab, self.tgt_vocab)

        self.data_train = self.apply_vocab(data_train)
        self.data_val = self.apply_vocab(data_val)
        self.data_test = self.apply_vocab(data_test)

    def read_file(self, fpath):
        # run_geo_pre = False
        # if 'geo' in fpath:
        #     logger.warning("Run special preprecossing for geo")
        #     run_geo_pre = True

        with open(fpath) as f:
            data = [json.loads(line) for line in f]

        converted = []
        for di, item in enumerate(data):
            question = item["question"]
            program = item["program"]

            # if run_geo_pre:
            #     assert program.startswith('answer')
            #     program = program[7:]

            # just drop brackets because in FunQL:
            # 1. we can recover them according to the grammar
            # 2. they do not tell much about spans
            program, spans = tokenize_program(program)
            spans = [[item[0], item[1]] for item in spans if item[1] > item[0] + 1]

            question = word_tokenize(question)
            assert len(program) > 1 and len(question) > 1

            inst = {
                "id": di,
                "src": question,
                "tgt": program,
                "tgt_spans": spans,
                "tgt_mask": self.gen_impossible_span_mask(spans, len(program)),
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
            masks.append(torch.tensor([span_mask[i, i + w + 1] for i in range(length - w)]))
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

    def add_eos(self, data):
        for item in data:
            item["src"].append("<eos>")
            item["tgt"].append("<eos>")
        return data

    def load_prior_alignment(self, data, path):
        # read probs generated by efmaral.
        # List[np.ndarray], each has shape (tgt_len x src_len)
        with open(path, "rb") as f:
            prior_alignments = pickle.load(f)
        assert len(data) == len(prior_alignments)
        for item, pa in zip(data, prior_alignments):
            pa[:, :-1] += pa[:, -1, None] / (pa.shape[1] - 1)
            pa = np.transpose(pa)[:-1]
            assert pa.shape[0] == len(item["src"])
            assert pa.shape[1] == len(item["tgt"])
            item["prior_alignment"] = pa
        return data

    def build_vocab(self, data):
        src_vocab_cnt = Counter()
        tgt_vocab_cnt = Counter()
        for inst in data:
            src_vocab_cnt.update(inst["src"])
            tgt_vocab_cnt.update(inst["tgt"])
        # if self.hparams.copy_mode != "none":
        #     logger.warning("I set src tokens in tgt vocab due to copy mode.")
        #     for inst in data:
        #         tgt_vocab_cnt.update(inst["src"])
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
        src_lens = [len(inst["src_ids"]) for inst in data]
        tgt_lens = [len(inst["tgt_ids"]) for inst in data]
        max_src_len = max(src_lens)
        max_tgt_len = max(tgt_lens)
        batched_src_ids = torch.full((len(tgt_lens), max_src_len), self.src_vocab.pad_token_id)
        batched_tgt_ids = torch.full((len(tgt_lens), max_tgt_len), self.tgt_vocab.pad_token_id)
        for i, item in enumerate(data):
            s, t = item["src_ids"], item["tgt_ids"]
            batched_src_ids[i, : len(s)] = torch.tensor(s)
            batched_tgt_ids[i, : len(t)] = torch.tensor(t)

        batched = {}
        batched["id"] = torch.tensor([inst["id"] for inst in data])
        batched["src"] = src
        batched["tgt"] = tgt
        batched["src_ids"] = batched_src_ids
        batched["tgt_ids"] = batched_tgt_ids
        batched["src_lens"] = src_lens
        batched["tgt_lens"] = tgt_lens

        copy_token = None
        if "copy_token" in data[0]:
            copy_token = torch.zeros(len(src), max_src_len, max_tgt_len, dtype=torch.bool)
            for i, item in enumerate(data):
                item = torch.from_numpy(item["copy_token"])
                copy_token[i, : item.shape[0], : item.shape[1]] = item
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
            offset_mapping = torch.zeros(transformer_inp["input_ids"].shape, dtype=torch.long)
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
        if "prior_alignment" in data[0]:
            prior_alignment = torch.zeros(len(src), max_src_len, max_tgt_len)
            for i, item in enumerate(data):
                item = torch.from_numpy(item["prior_alignment"]).to(torch.float32)
                prior_alignment[i, : item.shape[0], : item.shape[1]] = (
                    item * 0.9 + torch.ones_like(item) / item.shape[0] * 0.1
                )
            batched["prior_alignment"] = prior_alignment

        if "tgt_spans" in data[0]:
            batched["tgt_spans"] = [inst["tgt_spans"] for inst in data]

        if "tgt_mask" in data[0]:
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
            batched["tgt_masks"] = tgt_masks
        return batched


def tokenize_program(program):
    tokens = []
    spans = []
    buffer = []
    token_buffer = []
    comma = []
    single_quote = None

    for i, c in enumerate(program):
        if c in "()', ":
            if len(token_buffer) > 0:
                tokens.append("".join(token_buffer))
                token_buffer.clear()
            if c == "(":
                buffer.append(len(tokens))
                comma.append(len(tokens))
            elif c == ")":
                _c = comma.pop()
                _b = buffer.pop()
                if _c != _b:
                    spans.append((_c, len(tokens), "argument"))
                spans.append((_b, len(tokens), "all_argument"))
            elif c == "'":
                if single_quote is None:
                    single_quote = len(tokens)
                else:
                    spans.append((single_quote, len(tokens), "noun"))
                    single_quote = None
            elif c == ",":
                spans.append((comma[-1], len(tokens), "argument"))
                comma[-1] = len(tokens)
        else:
            token_buffer.append(c)
    assert single_quote is None and len(buffer) == 0
    return tokens, spans


if __name__ == "__main__":

    datamodule = SemanticParsingDataModule(
        "data/geo/funql/dev_len.json",
        "data/geo/funql/dev_len.json",
        "data/geo/funql/dev_len.json",
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
