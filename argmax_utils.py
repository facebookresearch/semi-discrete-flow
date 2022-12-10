"""
MIT License

Copyright (c) 2021 Didrik Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch


def integer_to_base(idx_tensor, base, dims):
    """
    Encodes index tensor to a Cartesian product representation.
    Args:
        idx_tensor (LongTensor): An index tensor, shape (...), to be encoded.
        base (int): The base to use for encoding.
        dims (int): The number of dimensions to use for encoding.
    Returns:
        LongTensor: The encoded tensor, shape (..., dims).
    """
    powers = base ** torch.arange(dims - 1, -1, -1, device=idx_tensor.device)
    floored = torch.div(idx_tensor[..., None], powers, rounding_mode="floor")
    remainder = floored % base

    base_tensor = remainder
    return base_tensor


def base_to_integer(base_tensor, base):
    """
    Decodes Cartesian product representation to an index tensor.
    Args:
        base_tensor (LongTensor): The encoded tensor, shape (..., dims).
        base (int): The base used in the encoding.
    Returns:
        LongTensor: The index tensor, shape (...).
    """
    dims = base_tensor.shape[-1]
    powers = base ** torch.arange(dims - 1, -1, -1, device=base_tensor.device)
    powers = powers[(None,) * (base_tensor.dim() - 1)]

    idx_tensor = (base_tensor * powers).sum(-1)
    return idx_tensor
