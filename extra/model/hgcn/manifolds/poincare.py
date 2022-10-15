"""Poincare ball manifold. Copy from https://github.com/HazyResearch/hgcn """
import torch
import torch.jit as jit
from typing import List, Optional

from .base import Manifold
from ..utils.math_utils import *
from ..utils import gyrovector_utils as gyro

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    @jit.script_method
    def sqdist(self, p1, p2, c, square: bool=True):
        sqrt_c = c ** 0.5
        dist_c = self.artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        if square:
            return dist ** 2
        else:
            return torch.abs(dist)

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    @jit.script_method
    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = self.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    @jit.script_method
    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * self.artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    @jit.script_method
    def mobius_add(self, x, y, c, dim: int=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def mobius_scalar_mul(r: torch.Tensor, x: torch.Tensor, c: torch.Tensor, dim: int = -1):
        x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
        res_c = tan_k(r * artan_k(x_norm, -c), ) * (x / x_norm)
        return res_c

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

    # def weighted_midpoint(
    #     self,
    #     xs: torch.Tensor,
    #     c: torch.Tensor,
    #     weights: Optional[torch.Tensor] = None,
    #     *,
    #     reducedim: Optional[List[int]] = None,
    #     dim: int = -1,
    #     keepdim: bool = False,
    #     lincomb: bool = False,
    #     posweight=False,
    #     project=True,
    # ):
    #     mid = gyro.weighted_midpoint(
    #         xs=xs,
    #         k=-c,
    #         weights=weights,
    #         reducedim=reducedim,
    #         dim=dim,
    #         keepdim=keepdim,
    #         lincomb=lincomb,
    #         posweight=posweight,
    #     )
    #     if project:
    #         return gyro.project(mid, k=-c, dim=dim)
    #     else:
    #         return mid

    # def weighted_midpoint_bmm(
    #     self,
    #     xs: torch.Tensor,
    #     c: torch.Tensor,
    #     weights: torch.Tensor,
    #     lincomb: bool = False,
    #     project=True,
    # ):
    #     mid = gyro.weighted_midpoint_bmm(
    #         xs=xs,
    #         weights=weights,
    #         k=-c,
    #         lincomb=lincomb,
    #     )
    #     if project:
    #         return gyro.project(mid, k=-c, dim=-1)
    #     else:
    #         return mid

    # def weighted_midpoint_spmm(
    #     self,
    #     xs: torch.Tensor,
    #     c: torch.Tensor,
    #     weights: torch.sparse.FloatTensor,
    #     lincomb: bool = False,
    # ):
    #     gamma = self._lambda_x(xs, c)
    #     denominator = torch.sparse.mm(weights, gamma - 1)
    #     nominator = torch.sparse.mm(weights, gamma * xs)
    #     two_mean = nominator / denominator.clamp_min(1e-10) ## instead of clamp_abs
    #     a_mean = two_mean / (1. + (1. - c * two_mean.pow(2).sum(dim=-1, keepdim=True)).sqrt())

    #     # debug code
    #     if torch.any(torch.isnan(a_mean)) or torch.any(torch.isinf(a_mean)):
    #         raise ValueError('The result of the calculation is is nan / inf.')

    #     if lincomb:
    #         alpha = weights.abs().sum(dim=-1, keepdim=True)
    #         a_mean = self.mobius_scalar_mul(alpha, a_mean, c, dim=-1)
    #     return self.proj(a_mean, c)

    def weighted_midpoint_spmm(
        self,
        xs: torch.Tensor,
        c: torch.Tensor,
        weights: torch.sparse.FloatTensor,
        lincomb: bool = False,
    ):
        return gyro.weighted_midpoint_spmm(xs, k=-c, weights=weights, lincomb=lincomb)

    @jit.script_method
    def sqdist_for_mat(self, p1: torch.Tensor, p2: torch.Tensor, c: torch.Tensor, square: bool=True):
        r"""The matrix version of sqdist to accelerate computation.
        """
        sqrt_c = c ** 0.5
        dist_c = self.artanh(
            sqrt_c * self.mobius_add_for_mat(-p1, p2, c).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        if square:
            return dist ** 2
        else:
            return torch.abs(dist)

    @jit.script_method
    def mobius_add_for_mat(self, x, y, c):
        x2 = x.pow(2).sum(dim=1, keepdim=True)  # B x 1
        y2 = y.pow(2).sum(dim=0, keepdim=True)  # 1 x I
        xy = x @ y                              # B x I
        num = torch.unsqueeze(1 + 2 * c * xy + c * y2, -1) @ torch.unsqueeze(x, 1) + torch.einsum('b,di->bid', torch.squeeze(1 - c * x2, 1), y)
        denom = torch.unsqueeze(1 + 2 * c * xy + c ** 2 * x2 @ y2, -1)
        return num / denom.clamp_min(self.min_norm) # B x I x D

    def dist_from_ori(self, x, c, square=True):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * x.norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        if square:
            return dist ** 2
        else:
            return torch.abs(dist)
    
    def component_inner(self, x, c, u, keepdim=False):
        return self.inner(x, c, u, keepdim=keepdim)

    @jit.script_method
    def artanh(self, x: torch.Tensor):
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)
        return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

    @jit.script_method
    def tanh(self, x: torch.Tensor):
        return x.clamp(-15, 15).tanh()