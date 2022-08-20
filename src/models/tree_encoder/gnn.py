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


class GeneralGNNForGumbel(GeneralGNN):
    def build_gnn_input(
        self, spans_inp: List[List[Tuple[int, int]]], x: List[List[torch.Tensor]]
    ):
        # spans: batch x nspans x 2
        # x: batch x seq_len x hidden

        spans, parents = [], []
        graphs = []
        for bidx, spans_inst in enumerate(spans_inp):
            vertices, edges = [], []
            s, p, mapping = spans2tree(spans_inst, return_mapping=True)
            inv_mapping = list(range(len(s)))  # s to spans_inst
            inv_mapping.sort(key=lambda x: mapping[x])

            spans.append(spans_inst)
            parents_inst = []
            for i, span in enumerate(spans_inst):
                _parent = p[mapping[i]]
                vertices.append(x[bidx][i])
                if _parent != -1:
                    parent_i = inv_mapping[p[mapping[i]]]

                    # p_span = spans_inst[parent_i]
                    # assert p_span[0] <= span[0] <= span[1] <= p_span[1]
                    # assert span != p_span
                    edges.append((i, parent_i))
                    edges.append((parent_i, i))
                else:
                    parent_i = -1

                parents_inst.append(parent_i)

            parents.append(parents_inst)
            graphs.append(Data(torch.stack(vertices, 0), torch.tensor(edges).T))

        return (
            Batch.from_data_list(graphs).to(x[0][0].device),
            {
                "spans": spans,
                "parents": parents,
                "length": [len(item) for item in spans],
            },
        )


if __name__ == "__main__":
    from torch_geometric.nn import GCN

    spans_inst = [(0, 6), (1, 3), (1, 2), (4, 5)]
    spans = [spans_inst, spans_inst]
    x = torch.randn(2, 7, 5)

    model = GeneralGNN(GCN(5, 3, 2, 11))
    features, spans = model(spans, x)
    print(spans)
