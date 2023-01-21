import logging
import math
import pickle
from collections import Counter, defaultdict
from functools import partial, reduce
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from nltk.tokenize import WordPunctTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src import is_under_debugger
from src.datamodules.components.vocab import Vocabulary
from src.datamodules.datamodule import _DataModule

from .components.sampler import BucketedSampler, ByLengthSampler, kmeans

logger = logging.getLogger(__file__)


class TSVDataModule(_DataModule):
    def __init__(
        self,
        # file
        train_file,
        dev_file,
        test_file,
        prior_alignment_file: str = None,
        load_gold_tree: bool = False,
        # preprocess
        max_src_len: int = 100,
        max_tgt_len: int = 100,
        copy_mode: str = "none",
        transformer_tokenizer_name: str = None,
        tokenize_tgt: bool = False,
        vocab_min_freq: int = 3,
        emphasize: bool = False,  # for StylePTB's AEM VEM
        # sampler
        batch_size: int = 64,
        eval_batch_size: int = 64,
        force_src_same_length: bool = False,
        use_double_length_bucket: bool = False,
        double_length_bucket_rate: bool = 1.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_nltk_tokenizer: bool = False,
        ###
        debug_1=False,  # set target sequence to even length.
        debug_2=False,
        **kwargs,
    ):
        assert copy_mode in ("none", "token", "phrase")
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.nltk_tokenizer = WordPunctTokenizer()

        with self.trace_persistent_variables():
            self.src_vocab: Optional[Vocabulary] = None
            self.tgt_vocab: Optional[Vocabulary] = None
            self.use_transformer_tokenizer = transformer_tokenizer_name is not None
            self._fix_dot_transformer_tokenizer = False
            if transformer_tokenizer_name is not None:
                extra_args = {}
                if "roberta" in transformer_tokenizer_name:
                    extra_args["add_prefix_space"] = True
                    self._fix_dot_transformer_tokenizer = True
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

        if self.hparams.load_gold_tree:
            data_train = self.load_gold_tree(data_train, self.hparams.train_file)
            data_val = self.load_gold_tree(data_val, self.hparams.dev_file)
            data_test = self.load_gold_tree(data_test, self.hparams.test_file)

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

        data_train = self.process_pair(data_train)
        data_val = self.process_pair(data_val)
        data_test = self.process_pair(data_test)

        self.src_vocab, self.tgt_vocab = self.build_vocab(data_train)
        logger.info(f"src vocab size: {len(self.src_vocab)}. tgt vocab size: {len(self.tgt_vocab)}")

        self.data_train = self.apply_vocab(data_train)
        self.data_val = self.apply_vocab(data_val)
        self.data_test = self.apply_vocab(data_test)

    def read_file(self, fpath):
        data = []
        for i, d in enumerate(open(fpath, "r")):
            inst = self.process_line(d)
            inst["id"] = i
            if len(inst["src"]) > 0 and len(inst["tgt"]) > 0:
                data.append(inst)
        return data

    def process_line(self, line: str):
        src, tgt = line.split("\t")
        if self.hparams.use_nltk_tokenizer:
            src = self.nltk_tokenizer.tokenize(src)
            tgt = self.nltk_tokenizer.tokenize(tgt)
        else:
            src = src.strip().split()
            tgt = tgt.strip().split()

        if self.hparams.emphasize:
            assert src[-2] == ";"
            *src, _, emp_word = src
            emp = [item == emp_word for item in src]
            assert sum(emp) >= 1, (src, emp_word)
            emp = [int(item) + 1 for item in emp]

        if len(src) == 1 or len(tgt) == 1:
            src = src + src
            tgt = tgt + tgt

        if self.hparams.debug_1:
            if len(tgt) % 2 == 1:
                tgt.append("<dbg1>")

        if self.hparams.debug_2:
            tgt = tgt[::-1]

        if self.hparams.emphasize:
            return {"src": src, "tgt": tgt, "emp": emp}
        else:
            return {"src": src, "tgt": tgt}

    def process_pair(self, data):
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

    def load_gold_tree(self, data, path):
        from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

        path = Path(path)
        reader = BracketParseCorpusReader(str(path.parents[0]), [path.with_suffix(".tb").name])
        trees = list(reader.parsed_sents())
        assert len(trees) == len(data)
        for item, tree in zip(data, trees):
            tree.collapse_unary(True)
            item["src_tree"] = tree
        return data

    def load_prior_alignment(self, data, path):
        baseline = 0.85
        with open(path, "rb") as f:
            prior_alignments = pickle.load(f)
        assert len(data) == len(prior_alignments)
        for item, pa in zip(data, prior_alignments):
            assert pa.shape[0] == len(item["src"])
            assert pa.shape[1] == len(item["tgt"])
            item["prior_alignment"] = np.clip(baseline - pa + 1, 1e-6, None)
        return data

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
        src_vocab = Vocabulary(src_vocab_cnt, threshold=self.hparams.vocab_min_freq)
        tgt_vocab = Vocabulary(tgt_vocab_cnt, threshold=self.hparams.vocab_min_freq)
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

    def train_dataloader(self, keep_order=False):
        if keep_order:
            shuffle = False
            collator = partial(self.collator, sort=False)
        else:
            shuffle = not is_under_debugger()
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

    def get_batch_sampler(self, dataset, phase):
        force_src_same_length = self.hparams.force_src_same_length or (phase != "train")
        if self.hparams.use_double_length_bucket:

            if force_src_same_length:
                length_bucket = defaultdict(list)
                for i, item in enumerate(dataset):
                    length_bucket[len(item["src"])].append(i)
                buckets = {}
                for length, ids in length_bucket.items():
                    costs = [
                        len((dataset[i]["tgt"]) * len(dataset[i]["src"])) ** self.hparams.double_length_bucket_rate
                        for i in ids
                    ]
                    num_clusters = math.ceil(math.log2(len(ids)))

                    if len(ids) == 1:
                        k = costs[0]
                        while k in buckets:
                            k += 1e-9
                        buckets[k] = ids
                    else:
                        _centroids, _clusters = kmeans(costs, num_clusters)
                        assert sum(len(item) for item in _clusters) == len(ids)
                        for k, c in zip(_centroids, _clusters):
                            while k in buckets:
                                k += 1e-9
                            buckets[k] = [ids[i] for i in c]

                assert len(reduce(lambda x, y: set(x) | set(y), buckets.values(), set())) == len(dataset)
                for c in buckets.values():
                    l = len(dataset[c[0]]["src"])
                    for i in c[1:]:
                        assert l == len(dataset[i]["src"])
            else:
                buckets = dict(
                    zip(
                        *kmeans(
                            [len((item["tgt"]) * len(item["src"])) ** 1.1 for item in dataset],
                            math.ceil(math.log2(len(dataset))),
                        )
                    )
                )
            size = self.hparams.batch_size if phase == "train" else self.hparams.eval_batch_size
            assert size > 100, f"Current size({size}) is too small."
            return BucketedSampler(buckets, size, shuffle=phase == "train" and not is_under_debugger())

        if force_src_same_length:
            size = self.hparams.batch_size if phase == "train" else self.hparams.eval_batch_size
            return ByLengthSampler([len(item["src"]) for item in dataset], size)

        return None  # fallback to instance samplers

    def collator(self, data, sort=True):
        if sort:
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
            batched["transformer_inputs"], batched["transformer_offset"] = self.make_transformer_input(src)
            if self.hparams.tokenize_tgt:
                (
                    batched["tgt_transformer_inputs"],
                    batched["tgt_transformer_offset"],
                ) = self.make_transformer_input(tgt)

        if "src_tree" in data[0]:
            trees = [item["src_tree"] for item in data]
            batched["src_tree"] = trees

        if "prior_alignment" in data[0]:
            prior_alignment = torch.ones(len(src), max_src_len, max_tgt_len)
            for i, item in enumerate(data):
                item = torch.from_numpy(item["prior_alignment"]).to(torch.float32)
                prior_alignment[i, : item.shape[0], : item.shape[1]] = item
            batched["prior_alignment"] = prior_alignment

        if "emp" in data[0]:
            emp = torch.zeros(len(src), max_src_len, dtype=torch.long)
            for i, item in enumerate(data):
                item = torch.tensor(item["emp"], dtype=torch.long)
                emp[i, : item.shape[0]] = item
            batched["emphasize"] = emp

        return batched

    def make_transformer_input(self, seq):
        transformer_inp = self.transformer_tokenizer(
            seq,
            padding=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping_raw = transformer_inp.pop("offset_mapping")
        offset_mapping = torch.zeros(transformer_inp["input_ids"].shape, dtype=torch.long)
        if self._fix_dot_transformer_tokenizer:
            space_id = 6
        for i, (mapping, ids) in enumerate(zip(offset_mapping_raw, transformer_inp["input_ids"])):
            cursor = 0
            for j, (item, id_item) in enumerate(zip(mapping, ids)):
                if self._fix_dot_transformer_tokenizer:
                    if id_item.item() == space_id:
                        continue
                if item[0] == item[1] == 0:
                    if j != 0:
                        break
                    cursor = 0
                elif item[0] == 0:
                    cursor += 1
                offset_mapping[i, j] = cursor
            assert cursor == len(seq[i])
        return transformer_inp, offset_mapping

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        excluded = []
        if "src_tree" in batch:
            trees = batch.pop("src_tree")
            excluded.append(("src_tree", trees))
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in excluded:
            batch[key] = value
        return batch


if __name__ == "__main__":

    datamodule = TSVDataModule(
        "data/StylePTB/ATP/train.tsv",
        "data/StylePTB/ATP/valid.tsv",
        "data/StylePTB/ATP/test.tsv",
        load_gold_tree=True,
        copy_mode="phrase",
        batch_size=2,
    )
    datamodule.setup()
    print("Loaded.")

    print("=" * 80)
    for batch in datamodule.train_dataloader():
        print(batch)
        break
