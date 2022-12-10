"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ["ActNorm1d", "ActNorm2d", "ConditionalAffine1d", "ConditionalAffine2d"]


class ActNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer("initialized", torch.tensor(0))

    def shape(self, x):
        raise NotImplementedError

    def axis(self, x):
        return NotImplementedError

    def forward(self, x, *, logp=None, **kwargs):
        c = x.shape[self.axis(x)]

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, self.axis(x)).reshape(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape(x)).expand_as(x)
        weight = self.weight.view(*self.shape(x)).expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logp is None:
            return y
        else:
            return y, logp - self._logdetgrad(x)

    def inverse(self, y, logp=None, **kwargs):
        assert self.initialized
        bias = self.bias.view(*self.shape(y)).expand_as(y)
        weight = self.weight.view(*self.shape(y)).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logp is None:
            return x
        else:
            return x, logp + self._logdetgrad(x)

    def _logdetgrad(self, x):
        weight = self.weight.view(*self.shape(x)).expand(*x.size())
        return weight.reshape(x.shape[0], -1).sum(1, keepdim=True)

    def __repr__(self):
        return "{name}({num_features})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class ActNorm1d(ActNormNd):
    def shape(self, x):
        return [1] * (x.ndim - 1) + [-1]

    def axis(self, x):
        return x.ndim - 1


class ActNorm2d(ActNormNd):
    def shape(self, x):
        return [1, -1, 1, 1]

    def axis(self, x):
        return 1


class ConditionalAffineNd(nn.Module):
    def __init__(self, num_features, nnet):
        super(ConditionalAffineNd, self).__init__()
        self.num_features = num_features
        self.nnet = nnet

    def shape(self, x):
        raise NotImplementedError

    def axis(self, x):
        return NotImplementedError

    def _get_params(self, x, cond):
        f = self.nnet(cond.reshape(x.shape[0], -1))
        t = f[:, : self.num_features]
        s = f[:, self.num_features :]

        s = torch.sigmoid(s) * 0.98 + 0.01

        t = t.reshape(*self.shape(x)).expand_as(x)
        s = s.reshape(*self.shape(x)).expand_as(x)

        s = torch.sigmoid(s) * 0.98 + 0.01

        return t, s

    def forward(self, x, *, logp=None, cond=None, **kwargs):
        assert cond is not None, "This module only works when cond is provided."

        # Ehhh...
        if cond.ndim == 4:
            cond = cond[:, :, 0, 0]

        t, s = self._get_params(x, cond)
        y = x * s + t * (1 - s)

        if logp is None:
            return y
        else:
            logpy = logp - self._logdetgrad(s)
            return y, logpy

    def inverse(self, y, logp=None, cond=None, **kwargs):
        if cond.ndim == 4:
            cond = cond[:, :, 0, 0]

        t, s = self._get_params(y, cond)
        x = (y - t * (1 - s)) / s

        if logp is None:
            return x
        else:
            return x, logp + self._logdetgrad(s)

    def _logdetgrad(self, s):
        log_s = torch.log(s)
        return log_s.reshape(log_s.shape[0], -1).sum(1, keepdim=True)

    def __repr__(self):
        return "{name}({num_features})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class ConditionalAffine1d(ConditionalAffineNd):
    def shape(self, x):
        return [x.shape[0]] + [1] * (x.ndim - 2) + [-1]

    def axis(self, x):
        return x.ndim - 1


class ConditionalAffine2d(ConditionalAffineNd):
    def shape(self, x):
        return [x.shape[0], -1, 1, 1]

    def axis(self, x):
        return 1