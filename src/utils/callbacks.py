import io
import logging
import os
import pprint
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from rich.theme import Theme
from tqdm import tqdm

import wandb
from src.utils.log_utils import rich_theme

log = logging.getLogger("callback")


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


class CustomProgressBar(TQDMProgressBar):
    """Only one, short, ascii"""

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Validation sanity check",
            position=self.process_position,
            disable=self.is_disabled,
            leave=False,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=self.process_position,
            disable=self.is_disabled,
            leave=True,
            smoothing=0,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = tqdm(disable=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Testing",
            position=self.process_position,
            disable=self.is_disabled,
            leave=True,
            smoothing=0,
            ncols=0,
            ascii=True,
            file=sys.stdout,
        )
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"[{trainer.current_epoch + 1}] train")

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"[{trainer.current_epoch + 1}] val")

    def print(
        self,
        *args,
        sep: str = " ",
        end: str = os.linesep,
        file: Optional[io.TextIOBase] = None,
        nolock: bool = False,
    ):
        _info(sep.join(map(str, args)))
        # active_progress_bar = None
        #
        # if self.main_progress_bar is not None and not self.main_progress_bar.disable:
        #     active_progress_bar = self.main_progress_bar
        # elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
        #     active_progress_bar = self.val_progress_bar
        # elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
        #     active_progress_bar = self.test_progress_bar
        # elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
        #     active_progress_bar = self.predict_progress_bar
        #
        # if active_progress_bar is not None:
        #     s = sep.join(map(str, args))
        #     active_progress_bar.write(s, end=end, file=file, nolock=nolock)


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("console_kwargs", {"theme": rich_theme})
        super().__init__(*args, **kwargs)

    def print(
        self,
        *args,
        sep: str = " ",
        end: str = os.linesep,
        file: Optional[io.TextIOBase] = None,
        nolock: bool = False,
    ):
        _info(sep.join(map(str, args)))


class CustomWandbLogger(WandbLogger):
    def finalize(self, status: str) -> None:
        for fname in ["predict_on_test.txt", "train.log", "test.log"]:
            if os.path.exists(fname):
                wandb.save(fname)
        return super().finalize(status)
