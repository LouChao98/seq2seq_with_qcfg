import logging
from collections import Counter
from contextlib import contextmanager
from enum import IntEnum

import numpy as np

from src.datamodules.components.vocab import Vocabulary, VocabularyPair

from .tsv_datamodule import TSVDataModule

logger = logging.getLogger(__file__)


class TSV2DirDataModuleMode(IntEnum):
    NORMAL = 0
    INVERSE = 1


class TSV2DirDataModule(TSVDataModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        # vocab_pair is not traced because I overwrite __setattr__
        with self.trace_persistent_variables():
            self.vocab_pair_normal = None
            self.vocab_pair_inverse = None
        self._runtime_mode = TSV2DirDataModuleMode.NORMAL

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
        inst["copy_token_s2t"] = copy_m
        inst["copy_token_t2s"] = copy_m.transpose()

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
        inst["copy_phrase_s2t"] = output

        src = inst["tgt"]
        tgt = inst["src"]
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
        inst["copy_phrase_t2s"] = output

    def build_vocab(self, data):
        src_vocab_cnt = Counter()
        tgt_vocab_cnt = Counter()
        for inst in data:
            src_vocab_cnt.update(inst["src"])
            tgt_vocab_cnt.update(inst["tgt"])
        if self.hparams.copy_mode != "none":
            logger.warning("I copy tokens in vocabs to each other due to copy mode.")
            for inst in data:
                tgt_vocab_cnt.update(inst["src"])
                src_vocab_cnt.update(inst["tgt"])
        src_vocab = Vocabulary(src_vocab_cnt)
        tgt_vocab = Vocabulary(tgt_vocab_cnt)
        return src_vocab, tgt_vocab

    def collator(self, data):
        # ONLY SUPPORT QCFG
        assert not self.use_transformer_tokenizer, "not implemented"

        view1 = []
        view2 = []
        for item in data:
            view1_item, view2_item = {}, {}
            for key, value in item.items():
                if "src" in key:
                    view1_item[key] = value
                    view2_item[key.replace("src", "tgt")] = value
                elif "tgt" in key:
                    view1_item[key] = value
                    view2_item[key.replace("tgt", "src")] = value
                elif "_s2t" in key:
                    view1_item[key.replace("_s2t", "")] = value
                elif "_t2s" in key:
                    view2_item[key.replace("_t2s", "")] = value
                elif key in ("id",):
                    view1_item[key] = value
                    view2_item[key] = value
                else:
                    raise NotImplementedError
            view1.append(view1_item)
            view2.append(view2_item)

        batch1 = super().collator(view1, sort=False)
        batch2 = super().collator(view2, sort=False)
        return batch1, batch2

    @contextmanager
    def normal_mode(self):
        self._runtime_mode = TSV2DirDataModuleMode.NORMAL
        yield

    @contextmanager
    def inverse_mode(self):
        self._runtime_mode = TSV2DirDataModuleMode.INVERSE
        yield
        self._runtime_mode = TSV2DirDataModuleMode.NORMAL

    def __setattr__(self, key, value):
        if key == "vocab_pair":
            if value is None:
                return
            assert isinstance(value, VocabularyPair)
            self.__dict__["vocab_pair_normal"] = value
            self.__dict__["vocab_pair_inverse"] = value.swap()
        else:
            super().__setattr__(key, value)

    @property
    def vocab_pair(self):
        if self._runtime_mode == TSV2DirDataModuleMode.NORMAL:
            return self.vocab_pair_normal
        else:
            return self.vocab_pair_inverse
