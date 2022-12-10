"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        w_init = torch.randn(dim, dim)
        w_init = np.linalg.qr(w_init.numpy())[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, **kwargs):
        logp = kwargs.pop("logp", None)
        input_mask = kwargs.pop("input_mask", None)

        # Only apply to those with full input masks.
        if input_mask is None:
            input_mask = torch.as_tensor(True).expand_as(x)
        input_mask = input_mask.all(dim=-1, keepdim=True)

        y = F.linear(x, self.weight)
        y = y * input_mask + x * ~input_mask
        if logp is None:
            return y
        else:
            return y, logp - self._logdetgrad(x, input_mask)

    def inverse(self, y, **kwargs):
        logp = kwargs.pop("logp", None)
        input_mask = kwargs.pop("input_mask", None)
        input_mask = input_mask.all(dim=-1, keepdim=True)
        x = F.linear(y, self.weight.double().inverse().float())
        x = x * input_mask + y * ~input_mask
        if logp is None:
            return x
        else:
            return x, logp + self._logdetgrad(x, input_mask)

    def _logdetgrad(self, x, input_mask):
        nreps = input_mask.reshape(input_mask.shape[0], -1).sum(1, keepdim=True)
        return torch.slogdet(self.weight)[1] * nreps

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class InvertibleConv2d(nn.Module):

    def __init__(self, dim):
        super(InvertibleConv2d, self).__init__()
        self.dim = dim
        w_init = torch.randn(dim, dim)
        w_init = np.linalg.qr(w_init.numpy())[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, **kwargs):
        logp = kwargs.pop("logp", None)
        y = F.conv2d(x, self.weight.view(self.dim, self.dim, 1, 1))
        if logp is None:
            return y
        else:
            return y, logp - self._logdetgrad * x.shape[2] * x.shape[3]

    def inverse(self, y, **kwargs):
        logp = kwargs.pop("logp", None)
        x = F.conv2d(y, self.weight.inverse().view(self.dim, self.dim, 1, 1))
        if logp is None:
            return x
        else:
            return x, logp + self._logdetgrad * x.shape[2] * x.shape[3]

    @property
    def _logdetgrad(self):
        return torch.slogdet(self.weight)[1]

    def extra_repr(self):
        return 'dim={}'.format(self.dim)
