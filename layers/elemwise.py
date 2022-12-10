"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroMeanTransform(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logp=None):
        x = x - .5
        if logp is None:
            return x
        return x, logp

    def inverse(self, y, logp=None):
        y = y + .5
        if logp is None:
            return y
        return y, logp


class Normalize(nn.Module):

    def __init__(self, mean, std):
        nn.Module.__init__(self)
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x, logp=None):
        y = x.clone()
        c = len(self.mean)
        y[:, :c].sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        if logp is None:
            return y
        else:
            return y, logp - self._logdetgrad(x)

    def inverse(self, y, logp=None):
        x = y.clone()
        c = len(self.mean)
        x[:, :c].mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])
        if logp is None:
            return x
        else:
            return x, logp + self._logdetgrad(x)

    def _logdetgrad(self, x):
        logdetgrad = (
            self.std.abs().log().mul_(-1).view(1, -1, 1, 1).expand(x.shape[0], len(self.std), x.shape[2], x.shape[3])
        )
        return logdetgrad.reshape(x.shape[0], -1).sum(-1, keepdim=True)


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = (sigmoid(x) - a) / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=1e-6):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logp=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = safe_log(s) - safe_log(1 - s)
        if logp is None:
            return y
        return y, logp - self._logdetgrad(x)

    def inverse(self, y, logp=None):
        x = (torch.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logp is None:
            return x
        return x, logp + self._logdetgrad(x)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -safe_log(s - s * s) + math.log(1 - 2 * self.alpha)
        logdetgrad = logdetgrad.view(x.size(0), -1).sum(1, keepdim=True)
        return logdetgrad

    def __repr__(self):
        return ('{name}({alpha})'.format(name=self.__class__.__name__, **self.__dict__))


def safe_log(x):
	return torch.log(x.clamp(min=1e-22))


class Softplus(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x, logp=None):
        '''
        z = softplus(x) = log(1+exp(z))
        ldj = log(dsoftplus(x)/dx) = log(1/(1+exp(-x))) = log(sigmoid(x))
        '''
        z = F.softplus(x)
        ldj = F.logsigmoid(x).reshape(x.shape[0], -1).sum(1, keepdim=True)

        if logp is None:
            return z
        else:
            ldj = F.logsigmoid(x).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return z, logp - ldj

    def inverse(self, z, logp=None):
        '''x = softplus_inv(z) = log(exp(z)-1) = z + log(1-exp(-z))'''
        zc = z.clamp(self.eps)
        x = z + torch.log1p(-torch.exp(-zc))

        if logp is None:
            return x
        else:
            ldj = -F.logsigmoid(x).reshape(x.shape[0], -1).sum(1, keepdim=True)
            return x, logp - ldj