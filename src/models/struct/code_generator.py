# only support inputs with two tensors

from collections import defaultdict
from itertools import chain

einsum_expr = "abcd,adbf->acf"

inp, outp = einsum_expr.split("->")
inp1, inp2 = inp.split(",")

inp1_position = {"_S": 0}
for c in inp1:
    assert c not in inp1_position
    inp1_position[c] = len(inp1_position)

inp2_position = {"_S": 0}
for c in inp2:
    assert c not in inp2_position
    inp2_position[c] = len(inp2_position)

outp_position = {"_S": 0}
for c in outp:
    assert c not in outp_position
    outp_position[c] = len(outp_position)

to_reduce = list((set(inp1_position) | set(inp2_position)) - set(outp_position))

inp1_permutation = ["0"]
inp1_position_list = list(inp1_position.items())
inp1_position_list.sort(key=lambda x: x[1])
for c, p in inp1_position_list[1:]:
    if c not in to_reduce:
        inp1_permutation.append(str(p))
inp1_permutation += [str(inp1_position[c]) for c in to_reduce]

print(f"inp1 = inp1.permute({', '.join(inp1_permutation)})")
if len(to_reduce) > 1:
    print(f"inp1 = inp1.flatten({-len(to_reduce)})")
if (num_to_squeeze := len(set(inp1) - set(inp2))) > 0:
    print(f"inp1 = inp1[..., {', '.join(['None'] * num_to_squeeze)}, :]")

inp2_permutation = ["0"]
inp2_position_list = list(inp2_position.items())
inp2_position_list.sort(key=lambda x: x[1])
for c, p in inp2_position_list[1:]:
    if c not in to_reduce:
        inp2_permutation.append(str(p))
inp2_permutation += [str(inp2_position[c]) for c in to_reduce]

print(f"inp2 = inp2.permute({', '.join(inp2_permutation)})")
if len(to_reduce) > 1:
    print(f"inp2 = inp2.flatten({-len(to_reduce)})")
if (num_to_squeeze := len(set(inp2) - set(inp1))) > 0:
    print(f"inp2 = inp2[..., {', '.join(['None'] * num_to_squeeze)}, :]")
print("x = semiring.sum(semiring.mul(inp1, inp2), dim=-1)")
