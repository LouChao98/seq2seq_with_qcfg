import logging
import re
from copy import deepcopy
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate

log = logging.getLogger(__file__)


class ModelBase(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        optimizer_cfg = self.hparams.optimizer
        if optimizer_cfg.groups is None or len(optimizer_cfg.groups) == 0:
            params = self.parameters()
        else:
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.named_parameters():
                matches = [
                    i
                    for i, g in enumerate(optimizer_cfg.groups)
                    if re.match(g.pattern, name)
                ]
                if len(matches) > 1:
                    log.warning(
                        f"{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}"
                    )
                if len(matches) > 0:
                    log.debug(
                        f"{name} match {optimizer_cfg.groups[matches[0]].pattern}."
                    )
                    params[matches[0]].append(p)
                else:
                    log.debug(f"{name} match defaults.")
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    log.warning(f"Nothing matches {optimizer_cfg.groups[i].pattern}")
            params = [
                {"params": p, **optimizer_cfg.groups[i]}
                for i, p in enumerate(params)
                if len(p) > 0
            ]
            params.append({"params": default_group})

        optimizer = instantiate(optimizer_cfg.args, params=params, _convert_="all")

        if (scheduler_cfg := self.hparams.scheduler) is None:
            return optimizer

        scheduler_cfg = deepcopy(scheduler_cfg)
        steps_per_epoch = len(self.datamodule.train_dataloader())
        for key in scheduler_cfg.args:
            if isinstance(scheduler_cfg.args[key], str):
                epochs = re.match(r"(\d+) epochs?", scheduler_cfg.args[key])
                if epochs is not None:
                    scheduler_cfg.args[key] = steps_per_epoch * int(epochs.group(1))

        scheduler = instantiate(scheduler_cfg.args, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.interval,
                "frequency": scheduler_cfg.frequency,
                "monitor": scheduler_cfg.monitor,
                "strict": True,
            },
        }

