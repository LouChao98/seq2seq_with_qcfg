import logging

from .tsv_datamodule import TSVDataModule

logger = logging.getLogger(__file__)


class ArithmeticDataModule(TSVDataModule):
    COLNAME_TO_IDX = {"infix": 0, "prefix": 1, "postfix": 2}

    def __init__(
        self,
        src_column="infix",
        tgt_column="postfix",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        with self.trace_persistent_variables():
            self.src_column = self.COLNAME_TO_IDX[src_column]
            self.tgt_column = self.COLNAME_TO_IDX[tgt_column]
            assert self.src_column != self.tgt_column

    def process_line(self, line: str):
        cols = line.split("\t")
        src = list(cols[self.src_column].strip())
        tgt = list(cols[self.tgt_column].strip())
        inst = {"src": src, "tgt": tgt}
        return inst


if __name__ == "__main__":

    datamodule = ArithmeticDataModule(
        train_file="data/arithmetic/raw/train_10k.tsv",
        dev_file="data/arithmetic/raw/val_5k.tsv",
        test_file="data/arithmetic/raw/test_5k.tsv",
        copy_mode="none",
        batch_size=2,
    )
    datamodule.setup()
    print("Loaded.")

    print("=" * 80)
    for batch in datamodule.train_dataloader():
        print(batch)
        break
