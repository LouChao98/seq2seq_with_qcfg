import logging
from collections import Counter
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.datamodules.components.vocab import Vocabulary, VocabularyPair
from src.datamodules.datamodule import _DataModule
import numpy as np
from numba import jit

logger = logging.getLogger(__file__)


class TSVDataModule(_DataModule):
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
                self.transformer_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    transformer_tokenizer_name
                )

            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_train = self.read_file(self.hparams.train_file)
        data_train = [
            inst
            for inst in data_train
            if len(inst["src"]) <= self.hparams.max_src_len
            and len(inst["tgt"]) <= self.hparams.max_tgt_len
        ]
        data_val = self.read_file(self.hparams.dev_file)
        data_test = self.read_file(self.hparams.test_file)

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
        data = []
        for i, d in enumerate(open(fpath, "r")):
            inst = self.process_line(d)
            inst["id"] = i
            data.append(inst)
        return data

    def process_line(self, line: str):
        src, tgt = line.split("\t")
        src = src.strip().split()
        tgt = tgt.strip().split()
        if len(src) == 1 or len(tgt) == 1:
            src = src + src
            tgt = tgt + tgt
        inst = {"src": src, "tgt": tgt}
        return inst

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
        # NOTE this assume phrases are at least length 2
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
        loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        logger.info(f"Train dataloader: {len(loader)}")
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.eval_batch_size,
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
        src_lens = [len(inst["src_ids"]) for inst in data]
        argsort = list(range(len(data)))
        argsort.sort(key=lambda i: src_lens[i], reverse=True)
        data = [data[i] for i in argsort]

        src = [inst["src"] for inst in data]
        tgt = [inst["tgt"] for inst in data]
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

        return batched


if __name__ == "__main__":

    datamodule = TSVDataModule(
        "data/StylePTB/AEM/train.tsv",
        "data/StylePTB/AEM/valid.tsv",
        "data/StylePTB/AEM/test.tsv",
        enable_copy=True,
    )
    datamodule.setup()
    print("Loaded.")

    print("=" * 80)
    for batch in datamodule.train_dataloader():
        print(batch)
        break

