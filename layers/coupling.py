"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import ipdb
from functools import partial
import numpy as np
import torch
import torch.nn as nn

from .base.activations import Swish
from .mixlogcdf import mixture_log_cdf, mixture_log_pdf, mixture_inv_cdf
from .mixlogcdf import sigmoid, sigmoid_inverse, safe_log

__all__ = ["MaskedCouplingBlock", "MixLogCDFCouplingBlock", "coupling_block_fn"]


class MaskedCouplingBlock(nn.Module):
    """Coupling layer for images implemented using masks.

    Available mask types:
      - "channel0", "channel1" work on inputs with 2 or more axes and splits on the second axis..
      - "checkerboard0", "checkerboard1" only work on 4D inputs (BCHW).
    """

    def __init__(
        self, nnet, mask_dim=1, split_dim=-1, cond_dim=-1, mask_type="channel0"
    ):
        nn.Module.__init__(self)
        self.nnet = nnet
        self.mask_dim = (
            mask_dim  # used for constructing channel or skip masking patterns.
        )
        self.split_dim = (
            split_dim  # used for splitting the output into scale and shift.
        )
        self.cond_dim = cond_dim
        self.mask_type = mask_type

    def func_s_t(self, x, cond=None, **kwargs):
        split_dim = self.split_dim % x.ndim
        input_shape = x.shape

        if cond is not None:
            x = torch.cat([x, cond], dim=self.cond_dim)

        f = self.nnet(x)

        f = f.reshape(*input_shape[:split_dim], 2, *input_shape[split_dim:])
        s, t = f.split((1, 1), dim=split_dim)
        s = s.squeeze(split_dim)
        t = t.squeeze(split_dim)

        s = torch.sigmoid(s) * 0.98 + 0.01

        return s, t

    def forward(self, x, **kwargs):
        logp = kwargs.pop("logp", None)
        input_mask = kwargs.pop("input_mask", None)

        # get mask
        b = get_mask(x, dim=self.mask_dim, mask_type=self.mask_type).bool()

        # masked forward
        x_a = torch.where(b, x, torch.zeros_like(x))
        if input_mask is not None:
            x_a = torch.where(input_mask, x_a, torch.zeros_like(x_a))
        s, t = self.func_s_t(x_a, **kwargs)
        y = torch.where(~b, x * s + t * (1 - s), torch.zeros_like(x)) + x_a
        if input_mask is not None:
            y = torch.where(input_mask, y, torch.zeros_like(y))

        if logp is None:
            return y
        else:
            logpy = logp - self._logdetgrad(s, b, input_mask)
            return y, logpy

    def inverse(self, y, **kwargs):
        logp = kwargs.pop("logp", None)
        input_mask = kwargs.pop("input_mask", None)

        # get mask
        m = get_mask(y, dim=self.mask_dim, mask_type=self.mask_type).bool()

        # masked forward
        y_a = torch.where(m, y, torch.zeros_like(y))
        if input_mask is not None:
            y_a = torch.where(input_mask, y_a, torch.zeros_like(y_a))
        s, t = self.func_s_t(y_a, **kwargs)
        x = y_a + torch.where(
            ~m,
            (y - t * (1 - s)) / torch.where(~m, s, torch.ones_like(s)),
            torch.zeros_like(y),
        )
        if input_mask is not None:
            x = torch.where(input_mask, x, torch.zeros_like(x))

        if logp is None:
            return x
        else:
            logpx = logp + self._logdetgrad(s, m, input_mask)
            return x, logpx

    def _logdetgrad(self, s, mask, input_mask=None):
        masked_s = safe_log(torch.where(~mask, s, torch.ones_like(s)))
        if input_mask is not None:
            masked_s = torch.where(input_mask, masked_s, torch.zeros_like(masked_s))
        return masked_s.reshape(s.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return "mask_dim={mask_dim}, spit_dim={split_dim}, cond_dim={cond_dim}, mask_type={mask_type}".format(
            **self.__dict__
        )


class MixLogCDFCouplingBlock(nn.Module):
    """Coupling layer implemented using masks. Uses MixLogCDF as the dimension-wise transformation.
    """

    def __init__(
        self,
        nnet,
        num_mixtures,
        dim_size,
        mask_dim=1,
        split_dim=-1,
        mask_type="channel0",
    ):
        nn.Module.__init__(self)
        self.nnet = nnet
        self.num_mixtures = num_mixtures
        self.mask_dim = (
            mask_dim  # used for constructing channel or skip masking patterns.
        )
        self.split_dim = (
            split_dim  # used for splitting the output into scale and shift.
        )
        self.mask_type = mask_type

        self.a_rescale = nn.Parameter(torch.ones(dim_size))
        self.s_rescale = nn.Parameter(torch.ones(num_mixtures, dim_size))

    def func_params(self, x, cond=None, **kwargs):
        split_dim = self.split_dim % x.ndim
        input_shape = x.shape
        k = self.num_mixtures

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        f = self.nnet(x)

        f = f.reshape(*input_shape[:split_dim], -1, *input_shape[split_dim:])
        a, b, pi, mu, s = f.split((1, 1, k, k, k), dim=split_dim)
        a = torch.tanh(a.squeeze(split_dim)) * self.a_rescale.reshape(
            (-1,) + (1,) * (x.ndim - split_dim - 1)
        )
        b = b.squeeze(split_dim)
        s = torch.tanh(s) * self.s_rescale.reshape(
            (self.num_mixtures, -1) + (1,) * (x.ndim - split_dim - 1)
        )
        return a, b, pi, mu, s

    def forward(self, x, **kwargs):
        logp = kwargs.pop("logp", None)
        split_dim = self.split_dim % x.ndim

        # get mask
        m = get_mask(x, dim=self.mask_dim, mask_type=self.mask_type).bool()

        # masked forward
        x_a = torch.where(m, x, torch.zeros_like(x))

        a, b, pi, mu, s = self.func_params(x_a, **kwargs)

        out = mixture_log_cdf(x, pi, mu, s, dim=split_dim).exp()
        out, scale_ldj = sigmoid_inverse(out)
        out = (out + b) * a.exp()
        y = torch.where(~m, out, torch.zeros_like(out)) + x_a

        if logp is None:
            return y
        else:
            logistic_ldj = mixture_log_pdf(x, pi, mu, s, dim=split_dim)
            logpy = logp - self._logdetgrad(logistic_ldj + scale_ldj + a, m)
            return y, logpy

    def inverse(self, y, **kwargs):
        logp = kwargs.pop("logp", None)
        split_dim = self.split_dim % y.ndim

        # get mask
        m = get_mask(y, dim=self.mask_dim, mask_type=self.mask_type).bool()

        # masked forward
        y_a = torch.where(m, y, torch.zeros_like(y))

        a, b, pi, mu, s = self.func_params(y_a, **kwargs)

        out = y * a.mul(-1).exp() - b
        out, scale_ldj = sigmoid(out)
        out = out.clamp(1e-5, 1.0 - 1e-5)
        out = mixture_inv_cdf(out, pi, mu, s, dim=split_dim)
        logistic_ldj = mixture_log_pdf(out, pi, mu, s, dim=split_dim)
        x = torch.where(~m, out, torch.zeros_like(out)) + y_a

        if logp is None:
            return x
        else:
            logistic_ldj = mixture_log_pdf(x, pi, mu, s, dim=split_dim)
            logpx = logp + self._logdetgrad(logistic_ldj + scale_ldj + a, m)
            return x, logpx

    def _logdetgrad(self, s, mask):
        masked_s = torch.where(~mask, s, torch.zeros_like(s))
        return masked_s.reshape(s.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return (
            "num_mixtures={num_mixtures}, dim_size={dim_size}, "
            "mask_dim={mask_dim}, spit_dim={split_dim}, cond_dim={cond_dim}, mask_type={mask_type}"
        ).format(**self.__dict__)


def _get_checkerboard_mask(x, swap=False):
    n, c, h, w = x.size()

    H = ((h - 1) // 2 + 1) * 2  # H = h + 1 if h is odd and h if h is even
    W = ((w - 1) // 2 + 1) * 2

    # construct checkerboard mask
    if not swap:
        mask = torch.Tensor([[1, 0], [0, 1]]).repeat(H // 2, W // 2)
    else:
        mask = torch.Tensor([[0, 1], [1, 0]]).repeat(H // 2, W // 2)
    mask = mask[:h, :w]
    mask = mask.reshape(1, 1, h, w).expand(n, c, h, w).to(x)

    return mask


def _get_channel_mask(x, dim=1, swap=False):
    dim = dim % x.ndim
    c = x.shape[dim]
    if not swap:
        idx = (slice(None),) * dim + (slice(0, c // 2),)
    else:
        idx = (slice(None),) * dim + (slice(c // 2, None),)

    mask = torch.zeros_like(x)
    mask[idx] = 1
    return mask


def _get_skip_mask(x, dim=1, swap=False):
    dim = dim % x.ndim
    if not swap:
        idx = (slice(None),) * dim + (slice(0, None, 2),)
    else:
        idx = (slice(None),) * dim + (slice(1, None, 2),)

    mask = torch.zeros_like(x)
    mask[idx] = 1
    return mask


def get_mask(x, dim=1, mask_type=None):
    # --- Works on arbitrary tensors ---
    if mask_type is None:
        return torch.zeros(x.size()).to(x)
    elif mask_type == "channel0":
        return _get_channel_mask(x, dim=dim, swap=False)
    elif mask_type == "channel1":
        return _get_channel_mask(x, dim=dim, swap=True)
    elif mask_type == "skip0":
        return _get_skip_mask(x, dim=dim, swap=False)
    elif mask_type == "skip1":
        return _get_skip_mask(x, dim=dim, swap=True)
    # --- Only works on 4D inputs of BCHW ---
    elif mask_type == "checkerboard0":
        return _get_checkerboard_mask(x, swap=False)
    elif mask_type == "checkerboard1":
        return _get_checkerboard_mask(x, swap=True)
    else:
        raise ValueError("Unknown mask type {}".format(mask_type))


def coupling_block_fn(
    i,
    input_size,
    fc=False,
    idim=64,
    cond_embed_dim=0,
    zero_init=False,
    depth=4,
    actfn="softplus",
    mixlogcdf=False,
    num_mixtures=32,
    **kwargs,
):

    actfns = {
        "softplus": nn.Softplus,
        "swish": Swish,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }

    if fc or input_size[0] >= 4:
        mask_type = {0: "skip0", 1: "skip1", 2: "channel0", 3: "channel1",}[i % 4]
    else:
        mask_type = {0: "checkerboard0", 1: "checkerboard1"}[i % 2]

    if mixlogcdf:
        flow_fn = partial(
            MixLogCDFCouplingBlock, num_mixtures=num_mixtures, dim_size=input_size[0]
        )
        out_factor = 2 + 3 * num_mixtures
    else:
        flow_fn = MaskedCouplingBlock
        out_factor = 2

    # not sure if this actually works for fc=True
    layer_fn = nn.Linear if fc else partial(nn.Conv2d, kernel_size=3, padding=1)

    dim = np.prod(input_size) if fc else input_size[0]

    if depth > 1:
        in_dims = [dim + cond_embed_dim] + [idim] * (depth - 1)
        out_dims = [idim] * (depth - 1) + [dim * out_factor]
        layers = []
        for d_in, d_out in zip(in_dims, out_dims):
            layers.append(layer_fn(d_in, d_out))
            layers.append(actfns[actfn]())
        layers = layers[:-1]  # remove last actfn.
    else:
        layers = [layer_fn(dim + cond_embed_dim, dim * out_factor)]

    if zero_init:
        layers[-1]._layer.weight.data.fill_(0)

    net = nn.Sequential(*layers)
    return flow_fn(net, mask_dim=1, split_dim=1, mask_type=mask_type, **kwargs)
