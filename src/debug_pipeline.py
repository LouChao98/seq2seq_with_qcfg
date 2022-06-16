from copy import deepcopy
import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import pytorch_lightning as pl
from src import utils
from src.models.components.pcfg import PCFG

log = utils.get_logger(__name__)

def debug(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    pl.seed_everything(0)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup('fit')

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False)
    model.setup('fit', datamodule)
    if config.seed == 1:
        model.decoder.pcfg = PCFG()

    batch = next(iter(datamodule.train_dataloader()))
    model_output = model(batch)
    loss = model_output["decoder"] + model_output["encoder"]
    loss.backward()

    return
