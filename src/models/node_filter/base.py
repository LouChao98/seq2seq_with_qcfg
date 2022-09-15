import torch
from torch.distributions import Bernoulli


class NodeFilterBase(torch.nn.Module):
    def forward(self, spans, span_features, seq_features, **kwargs):
        ...

    def sample(self, gates, **kwargs):
        samples = []
        loglikelihood = []
        for gate in gates:
            dist = Bernoulli(probs=gate)
            sample = dist.sample() == 1
            samples.append(sample)
            loglikelihood.append((gate + 1e-9).log()[sample].sum())
        return samples, torch.stack(loglikelihood, dim=0)

    def argmax(self, gates, **kwargs):
        samples = []
        loglikelihood = []
        for gate in gates:
            sample = gate > 0.5
            samples.append(sample)
            loglikelihood.append((gate + 1e-9).log()[sample].sum())
        return samples, torch.stack(loglikelihood, dim=0)

    def apply_filter(self, samples, spans, span_features):
        # filtered_features = []
        filtered_spans = []
        for sample, span, feat in zip(samples, spans, span_features):
            span_ind = sample.nonzero(as_tuple=True)[0]
            filtered_spans.append([span[i] for i in span_ind])
            # filtered_features.append(feat[sample])
        return filtered_spans
