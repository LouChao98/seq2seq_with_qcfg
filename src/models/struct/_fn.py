import torch


def diagonal_copy_(x: torch.Tensor, y: torch.Tensor, b: int, w: int):
    assert x.is_contiguous()
    seq_len = x.size(2)
    stride = list(x.stride())
    new_stride = [stride[0], stride[1]]
    new_stride.append(stride[2] + stride[3])
    new_stride.extend(stride[4:])
    x.as_strided(
        size=(x.shape[0], b, seq_len - w, *list(x.shape[4:])),
        stride=new_stride,
        storage_offset=w * stride[3],
    ).copy_(y)


def stripe(x: torch.Tensor, b: int, n: int, w: int, offset=(0, 0), dim=1):
    assert x.is_contiguous()
    seq_len = x.size(3)
    stride = list(x.stride())
    numel = stride[3]
    stride[2] = (seq_len + 1) * numel
    stride[3] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(
        size=(x.shape[0], b, n, w, *list(x.shape[4:])),
        stride=stride,
        storage_offset=(offset[0] * seq_len + offset[1]) * numel,
    )


def stripe2(x: torch.Tensor, b: int, n: int, w: int, offset=(0, 0)):
    # x: c b seq_len width ...
    # x x x x  | 0 1 2 3
    # x 1 x x  | 4 5 6 7
    # 2 3 x x  | 8 9 0 1
    # 4 x x x  | 2 3 4 5
    # 5 8
    # 9 12
    assert x.is_contiguous()
    stride = list(x.stride())
    numel1, numel2 = stride[2], stride[3]
    stride[2] = numel1
    stride[3] = numel1 - numel2
    return x.as_strided(
        size=(x.shape[0], b, n, w, *list(x.shape[4:])),
        stride=stride,
        storage_offset=offset[0] * numel1 + offset[1] * numel2,
    )


def ns_diagonal_copy_(x: torch.Tensor, y: torch.Tensor, w):
    assert x.is_contiguous()
    seq_len = x.size(1)
    stride = list(x.stride())
    new_stride = [stride[0]]
    new_stride.append(stride[1] + stride[2])
    new_stride.extend(stride[3:])
    x.as_strided(
        size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
        stride=new_stride,
        storage_offset=w * stride[2],
    ).copy_(y)


def ns_diagonal(x: torch.Tensor, w: int):
    assert x.is_contiguous()
    x, seq_len = x.contiguous(), x.size(1)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.extend(stride[3:])
    return x.as_strided(
        size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
        stride=new_stride,
        storage_offset=w * stride[2],
    )


def ns_stripe(x: torch.Tensor, n: int, w: int, offset=(0, 0), dim=1):
    assert x.is_contiguous()
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(
        size=(x.shape[0], n, w, *list(x.shape[3:])),
        stride=stride,
        storage_offset=(offset[0] * seq_len + offset[1]) * numel,
    )
