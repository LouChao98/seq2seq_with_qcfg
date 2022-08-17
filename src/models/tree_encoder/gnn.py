from re import L
from typing import List, Tuple

import torch
from hydra.utils import instantiate
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GAT, GlobalAttention

from src.utils.fn import spans2tree


class GeneralGNN(torch.nn.Module):
    def __init__(self, nn, global_pooling, dim):
        # TODO: we can avoid instantiate if we can prevent interpolation
        # of in_channels at model's save_hyperparameters
        super().__init__()
        self.nn = instantiate(nn, in_channels=dim)
        self.global_pooling = instantiate(
            global_pooling, in_channels=self.nn.out_channels
        )

    def forward(self, x, lens, spans):
        # build graph
        batch, meta = self.build_gnn_input(spans, x)
        x = self.nn(batch.x, batch.edge_index)
        # split ~and pad~
        splitted = self.postprocess_gnn_output(x, meta)
        if self.global_pooling is not None:
            global_features = self.global_pooling(x, batch.batch, batch.edge_index)
            splitted = [
                localf + globalf.unsqueeze(0)
                for localf, globalf in zip(splitted, global_features)
            ]
        return splitted, meta["spans"]

    def build_gnn_input(self, spans_inp: List[List[Tuple[int, int]]], x: torch.Tensor):
        # spans: batch x nspans x 2
        # x: batch x seq_len x hidden

        spans, parents = [], []
        graphs = []
        for bidx, spans_inst in enumerate(spans_inp):
            vertices, edges = [], []
            spans_inst += [(i, i) for i in range(max(x[1] for x in spans_inst) + 1)]
            s, p = spans2tree(spans_inst)

            index = list(range(len(s)))
            index.sort(key=lambda x: (s[x][1] - s[x][0], s[x][0]))
            s = [s[i] for i in index]
            p = [p[i] for i in index]

            spans.append(s)
            parents.append(p)
            for i, (span, parent) in enumerate(zip(s, p)):
                # rep of span = start + end
                vertices.append(x[bidx, span[0]] + x[bidx, span[1]])
                if parent != -1:
                    edges.append((i, parent))
                    edges.append((parent, i))
            graphs.append(Data(torch.stack(vertices, 0), torch.tensor(edges).T))

        return (
            Batch.from_data_list(graphs).to(x.device),
            {
                "spans": spans,
                "parents": parents,
                "length": [len(item) for item in spans],
            },
        )

    def postprocess_gnn_output(self, x: torch.Tensor, meta):
        lens = meta["length"]
        x = torch.split(x, lens, 0)
        # x = pad_sequence(x, True)
        return x

    def get_output_dim(self):
        return self.nn.out_channels


if __name__ == "__main__":
    from torch_geometric.nn import GCN

    spans_inst = [(0, 6), (1, 3), (1, 2), (4, 5)]
    spans = [spans_inst, spans_inst]
    x = torch.randn(2, 7, 5)

    model = GeneralGNN(GCN(5, 3, 2, 11))
    features, spans = model(spans, x)
    print(spans)
