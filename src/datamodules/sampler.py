import logging
from functools import partial

import torch
from torch.utils.data import BatchSampler

log = logging.getLogger(__file__)


class ConstantTokenNumSampler(BatchSampler):
    def __init__(self,
                 seq_len,
                 max_token=4096,
                 max_sentence=-1,
                 num_bucket=-1,
                 single_sent_threshold=-1,
                 shuffle=True,
                 weight=None):
        """

        :param List[int] seq_len: list[int], 是每个sample的长度。一般可以通过dataset.get_field('seq_len').content传入
        :param int max_token: 每个batch的最大的token数量
        :param int single_sent_threshold: 长度大于阈值的句子强制batch size=1
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        :param bool shuffle: shuffle data each epoch. the order is not kept even shuffle=False due to bucket.
        """

        if len(seq_len) < num_bucket:
            log.warning("Too few samples. Batch size=1.")
            num_bucket = len(seq_len)
        assert num_bucket > 0

        self.seq_len = seq_len
        self.max_token = max_token
        self.max_sentence = max_sentence if max_sentence > 0 else 10000000000000000
        self.single_sent_threshold = single_sent_threshold
        self.shuffle = shuffle
        self.epoch = 0

        self.sizes, self.buckets = self.kmeans(seq_len, num_bucket)
        if weight is not None:
            for bucket in self.buckets:
                for i in range(len(bucket)):
                    bucket[i: i+1] *= weight[bucket[i]]
        self.chunks = [
            min(
                len(bucket),
                max(1, round(size * len(bucket) / max_token),
                    ((len(bucket) + max_sentence - 1) // max_sentence) if max_sentence >= 1 else 0))
            for size, bucket in zip(self.sizes, self.buckets)
        ]

        self.samples = sum(self.chunks) 
    
    def __len__(self):
        return self.samples

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
            g = torch.Generator()
            g.manual_seed(self.epoch)
            range_fn = partial(torch.randperm, generator=g)
        else:
            range_fn = torch.arange
        
        batches = []
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                batches.append([self.buckets[i][j] for j in batch.tolist()])
        
        if self.shuffle:
            batches = [batches[i] for i in range_fn(len(batches))]

        for batch in batches:
            yield from self._sort_in_batch(batch)


    def _sort_in_batch(self, batch):
        singles = []
        if self.single_sent_threshold != -1:
            new_batch = []
            for inst_idx in batch:
                if self.seq_len[inst_idx] >= self.single_sent_threshold:
                    singles.append([inst_idx])
                else:
                    new_batch.append(inst_idx)
            batch = new_batch
        batch.sort(key=lambda i: -self.seq_len[i])
        if len(batch):
            return [batch] + singles
        else:
            return singles

    @staticmethod
    def kmeans(x, k, max_it=32):
        """From https://github.com/yzhangcs/parser/blob/main/supar/utils/alg.py#L7"""

        # the number of clusters must not be greater than the number of datapoints
        x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
        # collect unique datapoints
        d = x.unique()
        # initialize k centroids randomly
        c = d[torch.randperm(len(d))[:k]]
        # assign each datapoint to the cluster with the closest centroid
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

        for _ in range(max_it):
            # if an empty cluster is encountered,
            # choose the farthest datapoint from the biggest cluster and move that the empty one
            mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
            while len(none) > 0:
                for i in none:
                    # the biggest cluster
                    b = torch.where(mask[mask.sum(-1).argmax()])[0]
                    # the datapoint farthest from the centroid of cluster b
                    f = dists[b].argmax()
                    # update the assigned cluster of f
                    y[b[f]] = i
                    # re-calculate the mask
                    mask = torch.arange(k).unsqueeze(-1).eq(y)
                none = torch.where(~mask.any(-1))[0].tolist()
            # update the centroids
            c, old = (x * mask).sum(-1) / mask.sum(-1), c
            # re-assign all datapoints to clusters
            dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
            # stop iteration early if the centroids converge
            if c.equal(old):
                break
        # assign all datapoints to the new-generated clusters
        # the empty ones are discarded
        assigned = y.unique().tolist()
        # get the centroids of the assigned clusters
        centroids = c[assigned].tolist()
        # map all values of datapoints to buckets
        clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

        return centroids, clusters
