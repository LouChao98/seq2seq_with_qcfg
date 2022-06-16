from re import L
from typing import List, Tuple
import torch

def spans2tree(spans: List[Tuple[int, int]]):
    spans.sort(key=lambda x: (x[0], -x[1]))
    parent = [i - 1 for i in range(len(spans))]
    endpoint = [-1 for i in range(len(spans))]

    prev_left = spans[0][0]
    prev_left_i = 0

    for i, span in enumerate(spans[1:], start=1):
        if span[0] == prev_left:
            continue
        endpoint[prev_left_i] = i - 1
        
        possible_parent_start = prev_left_i
        possible_parent_end = i - 1
        while spans[possible_parent_start][1] < span[1]:
            possible_parent_start = parent[possible_parent_start]
            possible_parent_end = endpoint[possible_parent_start]

        cursor = (possible_parent_start + possible_parent_end) // 2
        while possible_parent_end > possible_parent_start:
            if spans[cursor][1] < span[1]:
                possible_parent_start = cursor + 1
            else:
                possible_parent_end = cursor
        
        prev_left = span[0]
        prev_left_i = i
        parent[i] = possible_parent_start
    return spans, parent


def encode_with_gnn(spans: List[List[Tuple[int, int]]], x: torch.Tensor, nn):
    # spans: batch x nspans x 2
    # x: batch x seq_len x hidden

    spans, parents, vertices, edges = [], [], [], []
    offset = 0
    for bidx, spans_inst in enumerate(spans):
        s, p = spans2tree(spans_inst)
        spans.append(s)
        parents.append(p)

        for i, (span, parent) in enumerate(zip(spans, parents)):
            vertices.append(x[bidx, span[0]] + x[bidx, span[1]])
            if parent != -1:
                edges.append((offset + i, offset + parent))
                edges.append((offset + parent, offset + i))

        offset += len(s)




if __name__ == '__main__':
    spans = [[0, 6], [1, 3], [1,2], [4,5]]
    print(spans2tree(spans))