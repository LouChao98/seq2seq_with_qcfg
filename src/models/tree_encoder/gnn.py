from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import torch
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from hydra.utils import instantiate
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx

from src.models.components.allennlp_module import BucketedSpanExtractor
from src.utils.fn import spans2tree


class GeneralGNN(torch.nn.Module):
    def __init__(self, nn, global_pooling, dim, span_extractor=None):
        super().__init__()
        if span_extractor is not None:
            self.span_extractor: Union[BucketedSpanExtractor, SpanExtractor] = instantiate(
                span_extractor, input_dim=dim
            )
            self.nn = instantiate(nn, in_channels=self.span_extractor.get_output_dim())
        else:
            self.span_extractor = None
            self.nn = instantiate(nn, in_channels=dim)

        self.global_pooling = instantiate(global_pooling, in_channels=self.nn.out_channels)

    def forward(self, x, lens, spans):
        # build graph
        batch, meta = self.build_gnn_input(spans, x)
        x = self.nn(batch.x, batch.edge_index)
        # split
        splitted = self.postprocess_gnn_output(x, meta)
        if self.global_pooling is not None:
            global_features = self.global_pooling(x, batch.batch, edge_index=batch.edge_index)
            splitted = [localf + globalf.unsqueeze(0) for localf, globalf in zip(splitted, global_features)]
        return splitted, meta["spans"]

    def build_gnn_input(self, spans_inp: List[List[Tuple[int, int]]], x: torch.Tensor):
        # spans: batch x nspans x 2
        # x: batch x seq_len x hidden

        spans, parents = [], []
        graphs = []
        for bidx, spans_item in enumerate(spans_inp):
            vertices, edges = [], []
            parents_item = spans2tree(spans_item)
            spans.append(spans_item)
            parents.append(parents_item)

            if self.span_extractor is not None:
                span_indices = torch.tensor([(item[0], item[1] - 1) for item in spans_item])
                span_indices = span_indices.unsqueeze(0).to(x.device)
                vertices = self.span_extractor(x[bidx : bidx + 1], span_indices).squeeze(0)
                for i, (span, parent) in enumerate(zip(spans_item, parents_item)):
                    if parent != -1:
                        edges.append((i, parent))
                        edges.append((parent, i))
            else:
                for i, (span, parent) in enumerate(zip(spans_item, parents_item)):
                    # rep of span = start + end
                    vertices.append(x[bidx, span[0]] + x[bidx, span[1] - 1])
                    if parent != -1:
                        edges.append((i, parent))
                        edges.append((parent, i))
                    vertices = torch.stack(vertices, 0)

            graph = Data(vertices, torch.tensor(edges).T)
            graph.node_label = torch.tensor([(l, r) for l, r, *_ in spans_item])
            graphs.append(graph)

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
        return x

    def get_output_dim(self):
        return self.nn.out_channels

    @staticmethod
    def draw_graph(graph):
        from networkx.drawing.nx_pydot import graphviz_layout

        graph = to_networkx(graph, node_attrs=["node_label"])
        labels = nx.get_node_attributes(graph, "node_label")
        pos = graphviz_layout(graph, prog="dot")
        pos = {int(k): v for k, v in pos.items()}
        nx.draw(graph, labels=labels, pos=pos)
        plt.savefig("graph.png")


class GeneralGNN2(GeneralGNN):
    def forward(self, features):
        # build graph
        batch, meta = self.build_gnn_input(features)
        x = self.nn(batch.x, batch.edge_index)
        # split
        splitted = self.postprocess_gnn_output(x, meta)
        if self.global_pooling is not None:
            global_features = self.global_pooling(x, batch.batch, edge_index=batch.edge_index)
            splitted = [localf + globalf.unsqueeze(0) for localf, globalf in zip(splitted, global_features)]
        return splitted, meta["spans"]

    def build_gnn_input(self, features):
        # spans: batch x nspans x 2
        # x: batch x seq_len x hidden

        spans, parents = [], []
        graphs = []
        for bidx, features_item in enumerate(features):
            vertices, edges = [], []
            spans_item = list(features_item.keys())
            parents_item = spans2tree(spans_item)
            spans.append(spans_item)
            parents.append(parents_item)
            for i, (span, parent) in enumerate(zip(spans_item, parents_item)):
                vertices.append(features_item[span])
                if parent != -1:
                    edges.append((i, parent))
                    edges.append((parent, i))
            graph = Data(torch.stack(vertices, 0), torch.tensor(edges).T)
            graph.node_label = torch.tensor([(l, r) for l, r, *_ in spans_item])
            graphs.append(graph)

        return (
            Batch.from_data_list(graphs).to(x.device),
            {
                "spans": spans,
                "parents": parents,
                "length": [len(item) for item in spans],
            },
        )


if __name__ == "__main__":
    from torch_geometric.nn import GCN
    from torch_geometric.nn.aggr import GraphMultisetTransformer

    spans_inst = [(0, 7, -1), (1, 4, -1), (1, 3, -1), (4, 6, -1)]
    spans = [spans_inst[:], spans_inst[:]]
    x = torch.randn(2, 7, 5)
    # GCN(5, 3, 2, 11)
    model = GeneralGNN(
        {
            "_target_": "torch_geometric.nn.GCN",
            "in_channels": 5,
            "hidden_channels": 3,
            "num_layers": 2,
            "out_channels": 11,
        },
        global_pooling=None,
        dim=5,
    )
    features, spans = model(x, None, spans)
    print(spans)
