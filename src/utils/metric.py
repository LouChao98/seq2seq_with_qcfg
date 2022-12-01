import gc
import logging
from typing import Any, List

import torch
import torchmetrics
from torchmetrics import Metric

from src.utils.executor import Executor

logger = logging.getLogger(__file__)


class PerplexityMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, nll, lens):
        self.nll += nll.sum()
        self.cnt += sum(lens)

    def compute(self):
        return torch.exp(self.nll.float() / self.cnt).item()


class WholeSentMatchAccuracy(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.total += len(targets)
        self.correct += sum(" ".join(p) == " ".join(t) for p, t in zip(preds, targets))

    def compute(self):
        return (self.correct.float() / (self.total + 1e-9)).item()


class UnlabeledSpanF1Score(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for preds_inst, targets_inst in zip(preds, targets):
            preds_inst = set((i, j) for i, j, *_ in preds_inst if j > i + 1)
            targets_inst = set((i, j) for i, j, *_ in targets_inst if j > i + 1)
            self.tp += len(preds_inst.intersection(targets_inst))
            self.gold += len(targets_inst)
            self.pred += len(preds_inst)

    def compute(self):
        return {
            "f1": (200 * self.tp / (self.pred + self.gold + 1e-12)).item(),
            "r": (100 * self.tp / (self.gold + 1e-12)).item(),
            "p": (100 * self.tp / (self.pred + 1e-12)).item(),
        }


class MultiMetric(Metric):
    def __init__(self, **metrics):
        super().__init__()
        self._metrics = metrics
        for k, v in metrics.items():
            self.add_module(k, v)

    def update(self, preds, targets):
        for n, m in self._metrics.items():
            if isinstance(m, (torchmetrics.BLEUScore, torchmetrics.SacreBLEUScore)):
                m.update(
                    [" ".join(item) for item in preds],
                    [[" ".join(item)] for item in targets],
                )
            else:
                m.update(preds, targets)

    def compute(self):
        outputs = {}
        for n, m in self._metrics.items():
            output = m.compute()
            if isinstance(output, dict):
                for k, v in output.items():
                    outputs[n + "/" + k] = v.item() if isinstance(v, torch.Tensor) else v
            else:
                outputs[n] = output.item() if isinstance(output, torch.Tensor) else output
        return outputs


class DenotationMetric(Metric):
    # mainly from `span-based-sp`
    def __init__(self, executor: Executor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct_counts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_counts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_nonempty_counts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_nonempty_counts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.batch_counts = 0.0  # not used to compute metric. no need to sync.
        self.executor = executor

    def update(
        self,
        predictions: List[List[str]],
        gold_targets: List[List[str]],
        scenes: List[str],
        answers: List[str],
        questions: List[List[str]],
    ):
        self.total_counts += len(predictions)
        self.batch_counts += 1
        if self.batch_counts % 1000 == 0:  # collect garbage once in a while
            gc.collect()

        is_should_print = False
        is_printed = False
        for i, (predicted_tokens, scene, answer, question) in enumerate(zip(predictions, scenes, answers, questions)):

            gold_tokens = gold_targets[i] if gold_targets is not None else ["no_targets"]

            for predicted in predicted_tokens:
                denotation = self.executor.execute(" ".join(predicted), scene)
                if not denotation.startswith("error_parse:"):
                    break

            gold_answer = answer if answer is not None else self.executor.execute(" ".join(gold_tokens))
            if gold_answer != "[]":
                self.total_nonempty_counts += 1
            if gold_answer == denotation:
                self.correct_counts += 1
                if gold_answer != "[]":
                    self.correct_nonempty_counts += 1
            elif not is_printed and is_should_print:  # print errors but not too much
                logger.info("ques: {}".format(" ".join(question)))
                logger.info("pred: {}".format(" ".join(predicted)))
                logger.info("gold: {}".format(" ".join(gold_tokens)))
                logger.info("deno: {}".format(denotation))
                logger.info("answ: {}".format(gold_answer))
                logger.info()
                is_printed = True

    def compute(self):
        return {
            "den_acc": (self.correct_counts / (self.total_counts + 1e-6)).item(),
            "den_ne_acc": (self.correct_nonempty_counts / (self.total_nonempty_counts + 1e-6)).item(),
        }


class GeoDenotationMetric(DenotationMetric):
    def update(
        self,
        predictions: List[List[str]],
        gold_targets: List[List[str]],
    ):
        self.total_counts += len(predictions)
        self.batch_counts += 1
        if self.batch_counts % 1000 == 0:  # collect garbage once in a while
            gc.collect()

        is_should_print = False
        is_printed = False

        for prediction, target in zip(predictions, gold_targets):
            prediction = self.executor.recovery(prediction)
            target = self.executor.recovery(target)
            denotation = self.executor.execute(prediction)
            gold_answer = self.executor.execute(target)

            if gold_answer != "[]":
                self.total_nonempty_counts += 1
            if gold_answer == denotation:
                self.correct_counts += 1
                if gold_answer != "[]":
                    self.correct_nonempty_counts += 1
            elif not is_printed and is_should_print:  # print errors but not too much
                logger.info("ques: {}".format(target))
                logger.info("pred: {}".format(prediction))
                logger.info("deno: {}".format(denotation))
                logger.info("answ: {}".format(gold_answer))
                logger.info()
                is_printed = True


class KGeoDenotationMetric(DenotationMetric):
    def update(
        self,
        predictions: List[List[List[str]]],
        gold_targets: List[List[str]],
    ):
        self.total_counts += len(predictions)
        self.batch_counts += 1
        if self.batch_counts % 1000 == 0:  # collect garbage once in a while
            gc.collect()

        for prediction_cands, target in zip(predictions, gold_targets):
            target = self.executor.recovery(target)
            gold_answer = self.executor.execute(target)

            if gold_answer != "[]":
                self.total_nonempty_counts += 1

            for prediction in prediction_cands:
                prediction = self.executor.recovery(prediction)
                denotation = self.executor.execute(prediction)

                if gold_answer == denotation:
                    self.correct_counts += 1
                    if gold_answer != "[]":
                        self.correct_nonempty_counts += 1
                    break
