"""Copy from https://github.com/mil-tokyo/hyperbolic_nn_plusplus """
import torch

from typing import List, Optional


def sign(x):
    return torch.sign(x.sign() + 0.5)

def sabs(x, eps: float = 1e-15):
    #return x.abs().add_(eps)
    return x.abs().clamp_min(eps)

def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)

def drop_dims(tensor: torch.Tensor, dims: List[int]):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    seen: int = 0
    for d in dims:
        tensor = tensor.squeeze(d - seen)
        seen += 1
    return tensor

def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res