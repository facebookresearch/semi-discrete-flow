"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn

__all__ = ["AffineAutoregressive"]


class AffineAutoregressive(nn.Module):
    def __init__(self, nnet, split_dim=-1, cond_dim=-1):
        nn.Module.__init__(self)
        self.nnet = nnet
        self.split_dim = (
            split_dim  # used for splitting the output into scale and shift.
        )
        self.cond_dim = cond_dim

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

        s, t = self.func_s_t(x, **kwargs)
        y = x * s + t * (1 - s)

        if logp is None:
            return y
        else:
            logpy = logp - self._logdetgrad(s)
            return y, logpy

    def inverse(self, y, **kwargs):
        raise NotImplementedError

    def _logdetgrad(self, s):
        masked_s = safe_log(s)
        return masked_s.reshape(s.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return "split_dim={split_dim}, cond_dim={cond_dim}".format(**self.__dict__)


class Reorder(nn.Module):
    def __init__(self, dim, perm_dim=1):
        super().__init__()
        self.dim = dim
        self.perm_dim = perm_dim
        self.register_buffer("randperm", torch.randperm(dim))
        self.register_buffer("invperm", torch.argsort(self.randperm))

    def forward(self, x, logp, **kwargs):
        y = torch.index_select(x, self.perm_dim, self.randperm)
        if logp is None:
            return y
        else:
            return y, logp

    def inverse(self, y, logp, **kwargs):
        x = torch.index_select(y, self.perm_dim, self.invperm)
        if logp is None:
            return x
        else:
            return x, logp

    def extra_repr(self):
        return "dim={dim}, perm_dim={perm_dim}".format(**self.__dict__)


def safe_log(x):
    return torch.log(x.clamp(min=1e-22))
