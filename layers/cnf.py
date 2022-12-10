"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from re import X
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

from . import diffeq_layers
from .base.activations import Swish


def divergence_bf(f, y, training, **unused_kwargs):
    sum_diag = 0.0
    for i in range(f.shape[1]):
        retain_graph = training or i < (f.shape[1] - 1)
        sum_diag += (
            torch.autograd.grad(
                f[:, i].sum(), y, create_graph=training, retain_graph=retain_graph
            )[0]
            .contiguous()[:, i]
            .contiguous()
        )
    return sum_diag.contiguous()


def divergence_approx(f, y, training, e=None, **unused_kwargs):
    assert e is not None
    dim = f.shape[1]
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=training, retain_graph=training)[
        0
    ][:, :dim].contiguous()
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


class CNF(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(
        self,
        func,
        divergence_fn=divergence_approx,
        rtol=1e-5,
        atol=1e-5,
        method="dopri5",
        fast_adjoint=True,
        cond_dim=None,
        nonself_connections=False,
    ):
        super().__init__()
        self.func = func
        self.divergence_fn = divergence_fn
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.fast_adjoint = fast_adjoint
        self.cond_dim = cond_dim
        self.nonself_connections = nonself_connections

        self.nfe = 0

    def reset_nfe_ts(self):
        self.nfe = 0

    def forward(self, x, *, logp=None, cond=None, **kwargs):

        if cond is not None:
            assert self.cond_dim is not None

        e = torch.randn_like(x)

        options = {}
        adjoint_kwargs = {}
        adjoint_kwargs["adjoint_params"] = self.parameters()

        if logp is None:
            logp = torch.zeros(x.shape[0], 1, device=x.device)

        if cond is None:
            initial_state = (e, x, logp)
        else:
            initial_state = (e, x, cond, logp)

        if self.fast_adjoint:
            adjoint_kwargs["adjoint_options"] = {"norm": "seminorm"}

        solution = odeint_adjoint(
            self.diffeq,
            initial_state,
            torch.tensor([self.start_time, self.end_time]).to(x),
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
            options=options,
            **adjoint_kwargs,
        )
        if cond is None:
            _, y, logpy = tuple(s[-1] for s in solution)
        else:
            _, y, _, logpy = tuple(s[-1] for s in solution)

        if logp is None:
            return y
        else:
            return y, logpy

    def diffeq(self, t, state):
        self.nfe += 1

        if self.cond_dim is None:
            e, x, _ = state
        else:
            e, x, cond, _ = state

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)

            if self.cond_dim is None:
                inputs = x
            else:
                inputs = torch.cat([x, cond], dim=self.cond_dim)

            dx = self.func(t, inputs)

            if self.nonself_connections:
                dx_div = self.func(t, inputs, rm_nonself_grads=True)
            else:
                dx_div = dx

            # Use brute force trace for testing if 2D.
            dim = np.prod(x.shape[1:])
            if not self.training and dim <= 2:
                div = divergence_bf(dx_div, x, self.training)
            else:
                div = self.divergence_fn(dx_div, x, self.training, e=e)

        if not self.training:
            dx = dx.detach()
            div = div.detach()

        if self.cond_dim is None:
            return torch.zeros_like(e), dx, -div.reshape(-1, 1)
        else:
            return torch.zeros_like(e), dx, torch.zeros_like(cond), -div.reshape(-1, 1)

    def extra_repr(self):
        return f"method={self.method}, cond_dim={self.cond_dim}, rtol={self.rtol}, atol={self.atol}, fast_adjoint={self.fast_adjoint}"


def cnf_block_fn(
    i,
    input_size,
    fc=False,
    idim=64,
    zero_init=False,
    depth=4,
    actfn="softplus",
    cond_embed_dim=0,
    **kwargs,
):
    del i

    actfns = {
        "softplus": nn.Softplus,
        "swish": Swish,
    }

    layer_fn = diffeq_layers.ConcatLinear if fc else diffeq_layers.ConcatConv2d

    dim = np.prod(input_size) if fc else input_size[0]

    if depth > 1:
        in_dims = [dim + cond_embed_dim] + [idim] * (depth - 1)
        out_dims = [idim] * (depth - 1) + [dim]
        layers = []
        for d_in, d_out in zip(in_dims, out_dims):
            layers.append(layer_fn(d_in, d_out))
            layers.append(actfns[actfn]())
        layers = layers[:-1]  # remove last actfn.
    else:
        layers = [layer_fn(dim + cond_embed_dim, dim)]

    if zero_init:
        layers[-1]._layer.weight.data.fill_(0)

    net = diffeq_layers.SequentialDiffEq(*layers)

    return CNF(net, **kwargs)
