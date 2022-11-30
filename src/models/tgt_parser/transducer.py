from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..struct.decomp1_copy_as_t import Decomp1
from .base import TgtParserBase


class NeuralTransducer(TgtParserBase):
    # This assume a tokenizer in datamodule

    def __init__(self, **kwargs):
        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
        self.seq2seq_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        self.special_tokens = [f"[PT{i}]" for i in range(self.pt_states)]
        self.seq2seq_tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.seq2seq.resize_token_embeddings(len(self.seq2seq_tokenizer))


def make_seg_pairs(src: List[List[str]], tgt: List[List[str]], max_src_width: int, max_tgt_width: int):
    # Substrings should respect BPE. We will enumerate all possible substrings.
    # So it may difficult to use subwords directly.
    ...
