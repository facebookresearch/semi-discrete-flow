"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn

__all__ = ['SqueezeLayer']


class SqueezeLayer(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logp=None, **kwargs):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logp is None:
            return squeeze_x
        else:
            return squeeze_x, logp

    def inverse(self, y, logp=None, **kwargs):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logp is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logp


def unsqueeze(input, upscale_factor=2):
    return torch.pixel_shuffle(input, upscale_factor)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.reshape(batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    output = input_view.permute(0, 1, 3, 5, 2, 4)
    return output.reshape(batch_size, out_channels, out_height, out_width)
