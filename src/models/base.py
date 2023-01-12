import logging
import os
import re
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate

from .components.dynamic_hp import DynamicHyperParameter

log = logging.getLogger(__file__)


class ModelBase(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dynamic_cfg = {}

    def on_train_epoch_start(self) -> None:
        self.apply_dynamic_cfg()
        return super().on_train_epoch_start()

    # def on_train_batch_start(self, batch: Any, batch_idx: int):
    #     print(
    #         torch.cuda.max_memory_allocated() / 1024 / 1024,
    #         len(batch["src_lens"]),
    #         max(batch["src_lens"]),
    #         max(batch["tgt_lens"]),
    #     )
    #     torch.cuda.reset_peak_memory_stats()

    def add_dynamic_cfg(self, name, command):
        """name: <obj nevigation>|<cfg nevigation>"""
        if name in self._dynamic_cfg:
            log.warning(f"Overwriting {name} with {command}")
        self._dynamic_cfg[name] = DynamicHyperParameter(command)

    def apply_dynamic_cfg(self):
        params = {key: value.get(self.current_epoch) for key, value in self._dynamic_cfg.items()}
        for key, value in params.items():
            obj_nev, cfg_nev = key.split("|")
            o = self.hparams
            if len(obj_nev) > 0:
                for attr_name in obj_nev.split("."):
                    o = getattr(o, attr_name)
            s = o
            cfg_nev = cfg_nev.split(".")
            for k in cfg_nev[:-1]:
                s = s[k]
            s[cfg_nev[-1]] = value
        return params

    def configure_optimizers(self):
        optimizer_cfg = self.hparams.optimizer
        if optimizer_cfg.groups is None or len(optimizer_cfg.groups) == 0:
            params = [item for item in self.parameters() if item.requires_grad]
        else:
            params = [[] for _ in optimizer_cfg.groups]
            default_group = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                matches = [i for i, g in enumerate(optimizer_cfg.groups) if re.match(g.pattern, name)]
                if len(matches) > 1:
                    log.warning(f"{name} is ambiguous: {[optimizer_cfg.groups[m].pattern for m in matches]}")
                if len(matches) > 0:
                    log.debug(f"{name} match {optimizer_cfg.groups[matches[0]].pattern}.")
                    params[matches[0]].append(p)
                else:
                    log.debug(f"{name} match defaults.")
                    default_group.append(p)
            for i in range(len(params)):
                if len(params[i]) == 0:
                    log.warning(f"Nothing matches {optimizer_cfg.groups[i].pattern}")
            params = [{"params": p, **optimizer_cfg.groups[i]} for i, p in enumerate(params) if len(p) > 0]
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

    def save_predictions(self, outputs, path=None):
        if path is None:
            path = "predict_on_test.txt"
            if os.path.exists(path):
                for i in range(2, 1000):
                    path = f"predict_on_test_{i}.txt"
                    if not os.path.exists(path):
                        break
        log.info(f"Writing to {os.path.abspath(path)}")
        if dist.is_initialized() and (ws := dist.get_world_size()):
            if len(self.datamodule.data_test) % ws != 0:
                # TODO should I warn when detect duplicates?
                log.warning(
                    "Do NOT report the above metrics because you are using"
                    "DDP and the size of the testset is odd, which means"
                    "there is one sample counted twice due to the"
                    "DistributedSampler. Run evaluation on the"
                    "predict_on_test.txt file"
                )
            merged = [None] * ws
            dist.all_gather_object(merged, outputs)
            if self.global_rank == 0:
                outputs = sum(merged, [])

        if self.global_rank == 0:
            preds = []
            for inst in outputs:
                preds_batch = inst["preds"]
                id_batch = inst["id"].tolist()
                preds.extend(zip(id_batch, preds_batch))
            preds.sort(key=lambda x: x[0])

            # remove duplicate
            to_remove = []
            for i in range(1, len(preds)):
                if preds[i - 1][0] == preds[i][0]:
                    to_remove.append(i)
            for i in reversed(to_remove):
                del preds[i]

            # check missing
            if not (preds[0][0] == 0 and preds[-1][0] == len(preds) - 1):
                # TODO for val, this warning may be raised due to filtering like length limitation.
                log.warning(f"There are some missing examples. Last id={preds[-1][0]}. Len={len(preds)}")

            with open(path, "w") as f:
                for id_, inst in preds:
                    # f.write(f"{id_}:\t")
                    f.write(" ".join(inst))
                    f.write("\n")

    def save_detailed_predictions(self, outputs, path=None):
        if path is None:
            path = "detailed_predict_on_test.txt"
            if os.path.exists(path):
                for i in range(2, 1000):
                    path = f"detailed_predict_on_test_{i}.txt"
                    if not os.path.exists(path):
                        break
        log.info(f"Writing to {os.path.abspath(path)}")
        if dist.is_initialized() and (ws := dist.get_world_size()):
            if len(self.datamodule.data_test) % ws != 0:
                log.warning(
                    "Do NOT report the above metrics because you are using"
                    "DDP and the size of the testset is odd, which means"
                    "there is one sample counted twice due to the"
                    "DistributedSampler. Run evaluation on the"
                    "predict_on_test.txt file"
                )
            merged = [None] * ws
            dist.all_gather_object(merged, outputs)
            if self.global_rank == 0:
                outputs = sum(merged, [])

        if self.global_rank == 0:
            preds = []
            for inst in outputs:
                preds_batch = inst["detailed"]
                id_batch = inst["id"].tolist()
                preds.extend(zip(id_batch, preds_batch))
            preds.sort(key=lambda x: x[0])

            # remove duplicate
            to_remove = []
            for i in range(1, len(preds)):
                if preds[i - 1][0] == preds[i][0]:
                    to_remove.append(i)
            for i in reversed(to_remove):
                del preds[i]

            # check missing
            if not (preds[0][0] == 0 and preds[-1][0] == len(preds) - 1):
                log.warning(f"There are some missing examples. Last id={preds[-1][0]}. Len={len(preds)}")

            with open(path, "w") as f:
                for id_, inst in preds:
                    # f.write(f"{id_}:\t")
                    f.write(">>> [Parse on gold target sequence] " + ">" * 33)
                    f.write("\n")
                    f.write(f"Score:\t{inst[1]}")
                    f.write("\n")
                    f.write(inst[0])
                    f.write("\n")
                    f.write("-" * 70)
                    f.write("\n")
                    f.write(f"Score:\t{inst[3]}")
                    f.write("\n")
                    f.write(inst[2])
                    f.write("\n")
                    f.write("<<< [Parse on predicted sequence] " + "<" * 35)
                    f.write("\n\n")
