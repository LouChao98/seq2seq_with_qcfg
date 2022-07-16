import torch


class Vocabulary:
    def __init__(self, count, add_unk=True, add_pad=True, threshold=1):
        self.count = count
        self.id2word = []
        self.unk_token_id = None
        self.pad_token_id = None
        self.threshold = threshold
        if add_unk:
            self.unk_token_id = len(self.id2word)
            self.id2word.append("<unk>")
        if add_pad:
            self.pad_token_id = len(self.id2word)
            self.id2word.append("<pad>")
        self.id2word += sorted(list(count.keys()))
        self.word2id = {word: i for i, word in enumerate(self.id2word)}

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, torch.Tensor):
            if ids.numel() == 1:
                return self.id2word[ids.item()]
            ids = ids.tolist()
        elif isinstance(ids, int):
            return self.id2word[ids]

        output = []
        for item in ids:
            output.append(self.convert_ids_to_tokens(item))
        return output

    def convert_tokens_to_ids(self, tokens, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        assert isinstance(tokens, (list, str)), f"Bad input: {tokens}"
        if isinstance(tokens, str):
            return (
                self.unk_token_id
                if self.count.get(tokens, 0) < threshold
                else self.word2id[tokens]
            )
        return [
            self.unk_token_id
            if self.count.get(token, 0) < threshold
            else self.word2id[token]
            for token in tokens
        ]

    @classmethod
    def from_file(cls, path, **kwargs):
        counts = {}
        with open(path) as f:
            for line in f:
                word, count = line.strip().split()
                counts[word] = int(count)
        return Vocabulary(counts, **kwargs)

    def __len__(self):
        return len(self.id2word)


class VocabularyPair:
    """Store src/tgt vocabularies and their id mappings.
    """

    def __init__(self, src: Vocabulary, tgt: Vocabulary):
        self.src = src
        self.tgt = tgt
        self.mapping = {}

        for i, w in enumerate(self.src.id2word):
            if w in self.tgt.word2id:
                self.mapping[i] = self.tgt.word2id[w]

    def src2tgt(self, src_ids):
        return [self.mapping[i] for i in src_ids]
