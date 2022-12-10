"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, **kwargs):
        logp = kwargs.pop("logp", None)
        if logp is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x, **kwargs)
            return x
        else:
            for i in range(len(self.chain)):
                x, logp = self.chain[i](x, logp=logp, **kwargs)
            return x, logp

    def inverse(self, y, **kwargs):
        logp = kwargs.pop("logp", None)
        if logp is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y, **kwargs)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logp = self.chain[i].inverse(y, logp=logp, **kwargs)
            return y, logp


class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, **kwargs):
        return self.flow.inverse(x, **kwargs)

    def inverse(self, y, **kwargs):
        return self.flow.forward(y, **kwargs)


class Lambda(nn.Module):

    def __init__(self, forward_fn, inverse_fn):
        super(Lambda, self).__init__()
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

    def forward(self, x, logp=None):
        y = self.forward_fn(x)
        if logp is None:
            return y
        else:
            return y, logp

    def inverse(self, y, logp=None):
        x = self.inverse_fn(y)
        if logp is None:
            return x
        else:
            return x, logp
