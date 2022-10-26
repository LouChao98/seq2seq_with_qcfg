import pickle

import fasttext

from src.datamodules.tsv_datamodule import TSVDataModule

datatmodule = TSVDataModule(
    train_file="data/StylePTB/ATP/train.tsv",
    dev_file="data/StylePTB/ATP/valid.tsv",
    test_file="data/StylePTB/ATP/test.tsv",
)
datatmodule.setup()
words = set(datatmodule.src_vocab.id2word) | set(datatmodule.tgt_vocab.id2word)
model = fasttext.load_model("data/fasttext/wiki-news-300d-1M-subword.vec")

vectors = {w: model.get_word_vector(w) for w in words}

with open("data/embedding/fasttext_styleptb_atp.pkl", "wb") as f:
    pickle.dump(vectors, f)
