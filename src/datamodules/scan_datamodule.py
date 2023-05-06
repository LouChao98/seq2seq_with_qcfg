import logging
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.sampler import ByLengthSampler
from src.datamodules.components.vocab import Vocabulary
from src.datamodules.datamodule import _DataModule

logger = logging.getLogger(__file__)


class SCANDataModule(_DataModule):
    def __init__(
        self,
        train_file,
        dev_file,
        test_file,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
        batch_size: int = 64,
        eval_batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        force_src_same_length: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        with self.trace_persistent_variables():
            self.src_vocab: Optional[Vocabulary] = None
            self.tgt_vocab: Optional[Vocabulary] = None

            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        data_train = self.read_file(self.hparams.train_file)
        data_val = self.read_file(self.hparams.dev_file)
        data_test = self.read_file(self.hparams.test_file)

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

        self.src_vocab, self.tgt_vocab = self.build_vocab(data_train)

        self.data_train = self.apply_vocab(data_train)
        self.data_val = self.apply_vocab(data_val)
        self.data_test = self.apply_vocab(data_test)

    def read_file(self, fpath):
        data = []
        for i, d in enumerate(open(fpath, "r")):
            src, tgt = d.split("IN: ")[1].split(" OUT: ")
            src = src.strip().split()
            tgt = tgt.strip().split()
            if len(src) == 1 or len(tgt) == 1:
                src = src + src
                tgt = tgt + tgt
            data.append({"src": src, "tgt": tgt, "id": i})
        return data

    def process_pair(self, data):
        return data

    def build_vocab(self, data):
        src_vocab_cnt = Counter()
        tgt_vocab_cnt = Counter()
        for inst in data:
            src_vocab_cnt.update(inst["src"])
            tgt_vocab_cnt.update(inst["tgt"])
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

    def get_batch_sampler(self, dataset, phase):
        force_src_same_length = self.hparams.force_src_same_length or (phase != "train")

        if force_src_same_length:
            size = self.hparams.batch_size if phase == "train" else self.hparams.eval_batch_size
            return ByLengthSampler([len(item["src"]) for item in dataset], size)

        return None  # fallback to instance samplers

    def train_dataloader(
        self,
    ):
        shuffle = True
        collator = self.collator
        batch_sampler = self.get_batch_sampler(self.data_train, "train")
        if batch_sampler is not None:
            loader = DataLoader(
                dataset=self.data_train,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collator,
            )
        else:
            loader = DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collator,
                shuffle=shuffle,
            )
        logger.info(f"Train dataloader: {len(loader)}")
        return loader

    def val_dataloader(self):
        batch_sampler = self.get_batch_sampler(self.data_val, "val")
        if batch_sampler is not None:
            loader = DataLoader(
                dataset=self.data_val,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collator,
            )
        else:
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
        batch_sampler = self.get_batch_sampler(self.data_test, "test")
        if batch_sampler is not None:
            loader = DataLoader(
                dataset=self.data_test,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collator,
            )
        else:
            loader = DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.eval_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collator,
                shuffle=False,
            )
        logger.info(f"Test dataloader: {len(loader)}")
        return loader

    def collator(self, data):
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

        return batched


if __name__ == "__main__":

    datamodule = SCANDataModule(
        "data/SCAN/tasks_train_length.txt",
        "data/SCAN/tasks_test_length.txt",
        "data/SCAN/tasks_test_length.txt",
    )
    datamodule.setup()
    print("Loaded.")

    print("=" * 80)
    for batch in datamodule.train_dataloader():
        print(batch)
        break
