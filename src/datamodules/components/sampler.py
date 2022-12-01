import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist


class BucketedSampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.
    Args:
        buckets (Dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]
        self.rank, self.n_replicas, self.n_samples = 0, 1, sum(self.n_batches)
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = sum(self.n_batches) // self.n_replicas + int(
                self.rank < sum(self.n_batches) % self.n_replicas
            )
        self.epoch = 1

        assert not distributed, "Need to review that whether this is a correct impl"

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        total, batches = 0, []
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = torch.arange if not self.shuffle else lambda x: torch.randperm(x, generator=g)
        for i, bucket in enumerate(self.buckets):
            split_sizes = [(len(bucket) - j - 1) // self.n_batches[i] + 1 for j in range(self.n_batches[i])]
            # DON'T use `torch.chunk` which may return wrong number of batches
            for batch in range_fn(len(bucket)).split(split_sizes):
                if total % self.n_replicas == self.rank:
                    batches.append([bucket[j] for j in batch.tolist()])
                total += 1
        self.epoch += 1
        return iter(batches[i] for i in range_fn(len(batches)).tolist())

    def __len__(self):
        return self.n_samples


def kmeans(x: List[int], k: int, max_it: int = 32) -> Tuple[List[float], List[List[int]]]:
    r"""
    KMeans algorithm for clustering the sentences by length.
    Args:
        x (List[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters, which is an approximate value.
            The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.
    Returns:
        List[float], List[List[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.
    Examples:
        >>> x = torch.randint(10, 20, (10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    x = torch.tensor(x, dtype=torch.float)
    # collect unique datapoints
    datapoints, indices, freqs = x.unique(return_inverse=True, return_counts=True)
    # the number of clusters must not be greater than the number of datapoints
    k = min(len(datapoints), k)
    # initialize k centroids randomly
    centroids = datapoints[torch.randperm(len(datapoints))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        for i in none:
            # the biggest cluster
            biggest = torch.where(mask[mask.sum(-1).argmax()])[0]
            # the datapoint farthest from the centroid of the biggest cluster
            farthest = dists[biggest].argmax()
            # update the assigned cluster of the farthest datapoint
            y[biggest[farthest]] = i
            # re-calculate the mask
            mask = torch.arange(k).unsqueeze(-1).eq(y)
        # update the centroids
        centroids, old = (datapoints * freqs * mask).sum(-1) / (freqs * mask).sum(-1), centroids
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(datapoints.unsqueeze(-1) - centroids).min(-1)
        # stop iteration early if the centroids converge
        if centroids.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = centroids[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(indices.unsqueeze(-1).eq(torch.where(y.eq(i))[0]).any(-1))[0].tolist() for i in assigned]

    return centroids, clusters


"""
Copy from https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/helper/data_module.py
---
Same as (Kim et al, 2019)
"""


class ByLengthSampler:
    def __init__(self, lengths, batch_size=4):
        self.group = defaultdict(list)
        self.seq_lens = lengths
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size = batch_size
        total = []

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for idx, lst in self.group.items():
            total = total + list(chunks(lst, self.batch_size))
        self.total = total

    def __iter__(self):
        random.shuffle(self.total)
        for batch in self.total:
            yield batch

    def __len__(self):
        return len(self.total)
