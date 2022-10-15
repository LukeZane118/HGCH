import torch
from torch import Tensor
from torch.nn.init import uniform_, constant_, xavier_uniform_

from extra.model.hgcn.manifolds.base import *

@torch.no_grad()
def get_uni_init_hyp_weights(tensor: Tensor, c: torch.nn.Parameter, manifold: Manifold, requires_grad: bool = True, scale: float = 0.1, padding_idx=None) -> ManifoldParameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return ManifoldParameter(manifold.expmap0(tensor, c), requires_grad, manifold, c)

@torch.no_grad()
def get_pop_init_hyp_weights(tensor: Tensor, pop: Tensor, c: torch.nn.Parameter, manifold: Manifold, requires_grad: bool = True, scale: float = 1., padding_idx=None) -> ManifoldParameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    bias = 1. - pop.min()
    # tensor.div_(pop + bias)
    # power-law
    tensor.div_((pop + bias).pow(1.1))
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return ManifoldParameter(manifold.expmap0(tensor, c), requires_grad, manifold, c)

@torch.no_grad()
def get_pop_xavier_init_hyp_weights(tensor: Tensor, pop: Tensor, c: torch.nn.Parameter, manifold: Manifold, requires_grad: bool = True, padding_idx=None) -> ManifoldParameter:
    # assert scale > 0
    xavier_uniform_(tensor)
    tensor.div_(pop)
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return ManifoldParameter(manifold.expmap0(tensor, c), requires_grad, manifold, c)

@torch.no_grad()
def get_uni_init_euc_weights(tensor: Tensor, requires_grad: bool = True, scale: float = 0.1, padding_idx=None) -> torch.nn.Parameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)

@torch.no_grad()
def get_pop_init_euc_weights(tensor: Tensor, pop: Tensor, requires_grad: bool = True, scale: float = 1., padding_idx=None) -> torch.nn.Parameter:
    assert scale > 0
    uniform_(tensor, -scale, scale)
    bias = 1. - pop.min()
    # tensor.div_(pop + bias)
    # power-law
    tensor.div_((pop + bias).pow(1.1))
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)

@torch.no_grad()
def get_pop_xavier_init_euc_weights(tensor: Tensor, pop: Tensor, requires_grad: bool = True, padding_idx=None) -> torch.nn.Parameter:
    # assert scale > 0
    xavier_uniform_(tensor)
    tensor.div_(pop)
    if padding_idx is not None:
        constant_(tensor[padding_idx], 0)
    return torch.nn.Parameter(tensor, requires_grad=requires_grad)