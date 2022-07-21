import torch
from torchmetrics import Metric
import torchmetrics


class PerplexityMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cnt", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, nll, lens):
        self.nll += nll.sum()
        self.cnt += sum(lens)

    def compute(self):
        return torch.exp(self.nll.float() / self.cnt)


class WholeSentMatchAccuracy(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.total += len(targets)
        self.correct += sum(" ".join(p) == " ".join(t) for p, t in zip(preds, targets))

    def compute(self):
        return self.correct.float() / (self.total + 1e-9)


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
            if isinstance(output, torch.Tensor):
                outputs[n] = output.item()
            else:
                for k, v in output:
                    outputs[n + "/" + k] = v
        return outputs
