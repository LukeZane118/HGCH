"""Math utils functions."""

import torch
import torch.jit as jit

from torch._six import with_metaclass

# from typing import List, Optional

from .utils import *

def cosh(x, clamp: float=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp: float=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp: float=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)


def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
            + 1382 / 155925 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x ** 3
    elif order == 2:
        return x + 1 / 3 * k * x ** 3 + 2 / 15 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


def artan_k(x: torch.Tensor, k: torch.Tensor):
    """
    k: Sectional curvature of manifold
    """
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)


def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
            - 1 / 11 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x ** 3
    elif order == 2:
        return x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5 - 1 / 7 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")


def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)
