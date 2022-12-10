"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import traceback
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InvertibleLogSoftmax", "InvertibleSoftmax", "ConditionalLogits", "LogitsToProbabilities", "LogitCenterShift"]


class InvertibleLogSoftmax(nn.Module):

    def __init__(self):
        super().__init__()
        # self.log_alpha = nn.Parameter(torch.zeros(1))
        self.register_buffer("log_alpha", torch.zeros(1))

    def forward(self, h, input_mask=None, logp=None):
        log_alpha = self.log_alpha.expand(*h.shape[:-1], 1)
        hp1 = torch.cat([h, log_alpha], dim=-1)
        logits = torch.log_softmax(hp1, dim=-1)

        if logp is None:
            return logits
        else:
            logdetjac = self._logdetjac(logits, input_mask)
            return logits, logp - logdetjac

    def inverse(self, logits, input_mask=None, logp=None):
        # assert torch.logsumexp(logits, -1).isclose(torch.zeros(logits.shape[:-1]).to(logits), atol=1e-4).all()
        log_alpha = self.log_alpha.expand(*logits.shape[:-1], 1)
        h = logits[..., :-1] - logits[..., -1].unsqueeze(-1) + log_alpha
        if logp is None:
            return h
        else:
            logdetjac = -self._logdetjac(logits, input_mask)
            return h, logp - logdetjac

    def _logdetjac(self, q, input_mask=None):
        logdiff = q[..., -1]
        if input_mask is not None:
            input_mask = torch.all(input_mask, dim=-1)
            logdiff = logdiff * input_mask
        return logdiff.reshape(q.shape[0], -1).sum(1, keepdim=True)


class InvertibleSoftmax(InvertibleLogSoftmax):
    """ Like `InvertibleLogSoftmax` but takes care of the normalization change due to exp."""

    def forward(self, z, logp=None):
        if logp is None:
            return torch.exp(super().forward(z))
        logits, logp = super().forward(z, logp=logp)
        probs = torch.exp(logits)
        logp = logp - logits[..., :-1].reshape(probs.shape[0], -1).sum(1, keepdim=True)
        return probs, logp

    def inverse(self, probs, logp=None):
        logits = stable_log(probs)
        if logp is None:
            return super().inverse(logits)
        z, logp = super().inverse(logits, logp=logp)
        logp = logp + logits[..., :-1].reshape(probs.shape[0], -1).sum(1, keepdim=True)

        return z, logp


class LogitsToProbabilities(nn.Module):

    def forward(self, logits, logp=None):
        probs = torch.exp(logits)
        if logp is None:
            return probs
        else:
            logp = logp - logits[..., :-1].reshape(probs.shape[0], -1).sum(1, keepdim=True)
            return probs, logp

    def inverse(self, probs, logp=None):
        logits = stable_log(probs)
        if logp is None:
            return logits
        else:
            logp = logp + logits[..., :-1].reshape(probs.shape[0], -1).sum(1, keepdim=True)
            return logits, logp


class ConditionalLogits(nn.Module):
    """Ensures the target dimension value is always greater than the other dimensions in the last axis.
    This is used to conditionally map to a corner of the probability simplex where the target probability is highest.
    """

    def forward(self, z, cond, input_mask=None, logp=None):
        K = z.shape[-1]
        idx = F.one_hot(cond, K + 1).bool()

        aug_z = torch.cat([z, torch.zeros(*z.shape[:-1], 1).to(z)], dim=-1)

        target_z = aug_z[idx].reshape(*z.shape[:-1], -1)
        other_z = aug_z[~idx].reshape(*z.shape[:-1], -1)
        max_z = torch.max(other_z, dim=-1)[0][..., None]

        updated_target_z = _logaddexp(target_z, max_z)

        logdetjac = target_z - updated_target_z

        new_z = torch.zeros_like(aug_z)
        new_z[idx] = updated_target_z.reshape(-1)
        new_z[~idx] = aug_z[~idx]
        new_z = new_z[..., :-1]

        # account for when cond is the last class.
        mask = (cond == K)
        masked_z = z[mask].reshape(-1, K)
        masked_z = -F.softplus(-masked_z)
        new_z[mask] = masked_z

        if logp is None:
            return new_z
        else:
            logdetjac = target_z - updated_target_z
            logdetjac[mask] = torch.log(1 - torch.sigmoid(z[mask]) + 1e-10).sum(-1, keepdim=True)
            if input_mask is not None:
                input_mask = torch.all(input_mask, dim=-1, keepdim=True)
                logdetjac = logdetjac * input_mask
            return new_z, logp - logdetjac.reshape(new_z.shape[0], -1).sum(1, keepdim=True)

    def inverse(self, z, cond, input_mask=None, logp=None):
        K = z.shape[-1]
        idx = F.one_hot(cond, K + 1).bool()

        aug_z = torch.cat([z, torch.zeros(*z.shape[:-1], 1).to(z)], dim=-1)

        target_z = aug_z[idx].reshape(*z.shape[:-1], -1)
        other_z = aug_z[~idx].reshape(*z.shape[:-1], -1)
        max_z = torch.max(other_z, dim=-1)[0][..., None]

        assert (target_z > max_z).all(), "Inverse is only applicable if target dim > max(other dims)"
        updated_target_z = _logsubexp(target_z, max_z)

        new_z = torch.zeros_like(aug_z)
        new_z[idx] = updated_target_z.reshape(-1)
        new_z[~idx] = aug_z[~idx]
        new_z = new_z[..., :-1]

        # account for when cond is the last class.
        mask = (cond == K)
        masked_z = z[mask]
        masked_new_z = -torch.log(torch.exp(-masked_z) - 1)
        new_z[mask] = masked_new_z

        if logp is None:
            return new_z
        else:
            logdetjac = target_z - updated_target_z
            logdetjac[mask] = (-z[mask] + masked_new_z).sum(-1, keepdim=True)
            if input_mask is not None:
                print(logdetjac.shape, input_mask.shape)
            return new_z, logp - logdetjac.reshape(new_z.shape[0], -1).sum(1, keepdim=True)


def _logaddexp(a, b):
    m = torch.max(a, b).detach()
    return torch.log(torch.exp(a - m) + torch.exp(b - m)) + m


def _logsubexp(a, b):
    m = torch.max(a, b).detach()
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


def jacobian(x, y):
    """Computes the Jacobian of a function f(x)=y given x and y.

    Assumes x.shape[:-1] == y.shape[:-1], and all but the last axis are independent.
    """
    jac = []
    with torch.enable_grad():
        ones = torch.ones_like(y[..., 0])
        for k in range(y.shape[-1]):
            jac.append(torch.autograd.grad(y[..., k], x, ones, create_graph=True)[0])
        return torch.stack(jac, dim=-2)


def stable_log(x):
    eps = torch.min(torch.min(x), torch.zeros(1).to(x)) + 1e-20
    if eps.item() < -1e-4:
        traceback.print_stack(file=sys.stdout)
        print(f"Taking the log of very negative value {eps.item()}")
    eps = eps.detach()
    return torch.log(x + eps)


class CenterShift(nn.Module):

    def __init__(self, alpha=1e-5):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x, logp=None):
        """Assumes x is on the probability simplex."""
        num_classes = x.shape[-1]
        c = torch.ones_like(x) / num_classes
        out = self.alpha * c + (1 - self.alpha) * x
        if logp is None:
            return out
        else:
            logp = logp - torch.log1p(-self.alpha) * torch.prod(torch.as_tensor(x.shape[1:])).reshape(-1, 1)
            return out, logp

    def inverse(self, x, logp=None):
        """Assumes x is on the probability simplex."""
        num_classes = x.shape[-1]
        c = torch.ones_like(x) / num_classes
        out = (x - self.alpha * c) / (1 - self.alpha)
        if logp is None:
            return out
        else:
            logp = logp + torch.log1p(-self.alpha) * torch.prod(torch.as_tensor(x.shape[1:])).reshape(-1, 1)
            return out, logp


class LogitCenterShift(nn.Module):

    def __init__(self, alpha=0.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, input_mask=None, logp=None):
        """Assumes x is log probability."""
        num_classes = x.shape[-1]
        c = torch.ones_like(x) / num_classes
        shiftx = _logaddexp(x + math.log(1 - self.alpha), torch.log(self.alpha * c))
        if logp is None:
            return shiftx
        else:
            logp = logp - self._logdetjac(x, shiftx, input_mask)
            return shiftx, logp

    def inverse(self, shiftx, input_mask=None, logp=None):
        """Assumes shiftx is log probability."""
        num_classes = shiftx.shape[-1]
        c = torch.ones_like(shiftx) / num_classes
        x = _logsubexp(shiftx - math.log(1 - self.alpha), torch.log(self.alpha / (1 - self.alpha) * c))
        if logp is None:
            return x
        else:
            logp = logp + self._logdetjac(x, shiftx, input_mask)
            return x, logp

    def _logdetjac(self, x, shiftx, input_mask=None):
        logdiff = (x[..., :-1] + math.log(1 - self.alpha) - shiftx[..., :-1])
        if input_mask is not None:
            logdiff = logdiff * input_mask
        return logdiff.reshape(x.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return 'alpha={alpha}'.format(**self.__dict__)


if __name__ == "__main__":

    print("--- Testing invertible softmax ---")

    f = InvertibleSoftmax()
    x = torch.randn(3, 10, requires_grad=True)
    logpx = torch.zeros(3, 1).to(x)
    q, logpq = f(x, logp=logpx)

    jac = jacobian(x, q[..., :-1])
    true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    print("Logdetjac error", torch.norm(true_logdetjac + logpq, p=float("inf")))

    x_recon, logpx_recon = f.inverse(q, logp=logpq)
    print("z recon error", torch.norm(x - x_recon, p=float("inf")))
    print("logp recon error", torch.norm(logpx - logpx_recon, p=float("inf")))
    print()

    # print("--- Testing invertible segment softmax ---")
    # f = InvertibleSegmentLogSoftmax(segment_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3]))
    # x = torch.randn(3, 9, requires_grad=True)
    # logpx = torch.zeros(3, 1).to(x)
    # q, logpq = f(x, logp=logpx)

    # jac = jacobian(x, q[..., f.indices][..., :-4])
    # true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    # print("Logdetjac error", torch.norm(true_logdetjac + logpq, p=float("inf")))

    # x_recon, logpx_recon = f.inverse(q, logp=logpq)
    # print("z recon error", torch.norm(x - x_recon, p=float("inf")))
    # print("logp recon error", torch.norm(logpx - logpx_recon, p=float("inf")))
    # print()

    print("--- Testing conditional logits ---")

    cond_transform = ConditionalLogits()
    z = torch.randn(5, 10, requires_grad=True)
    x = torch.tensor([0, 1, 2, 3, 10])
    logpz = torch.zeros(z.shape[:-1] + (1,)).to(z)
    upd_z, upd_logpz = cond_transform.forward(z, x, logp=logpz)

    logits = torch.log_softmax(torch.cat([upd_z, torch.zeros(5, 1)], dim=-1), dim=-1)
    tgt_mask = F.one_hot(x, 11).bool()
    tgt_logit = logits[tgt_mask].reshape(5)
    max_logit = torch.max(logits[~tgt_mask].reshape(5, -1), dim=-1)[0]
    print(tgt_logit - max_logit > 0)
    assert (tgt_logit - max_logit > 0).all()

    jac = jacobian(z, upd_z)
    true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    print("Logdetjac error", torch.norm(true_logdetjac + upd_logpz, p=float("inf")))

    z_recon, logpz_recon = cond_transform.inverse(upd_z, x, logp=upd_logpz)
    print("z recon error", torch.norm(z_recon - z, p=float("inf")))
    print("logp recon error", torch.norm(logpz_recon - logpz, p=float("inf")))
    print()

    print("--- Testing center shift ---")

    f = CenterShift(0.05)
    x = torch.randn(3, 10, requires_grad=True)
    x = torch.softmax(x, dim=-1)
    logpx = torch.zeros(x.shape[:-1] + (1,)).to(x)
    y, logpy = f(x, logp=logpx)

    jac = jacobian(x, y)
    true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    print("Logdetjac error", torch.norm(true_logdetjac + logpy, p=float("inf")))

    x_recon, logpx_recon = f.inverse(y, logp=logpy)
    print("z recon error", torch.norm(x - x_recon, p=float("inf")))
    print("logp recon error", torch.norm(logpx - logpx_recon, p=float("inf")))
    print()

    print("--- Testing logit center shift ---")

    f = LogitCenterShift(0.01)
    x = torch.randn(3, 10, requires_grad=True)
    x = torch.softmax(x, dim=-1)
    logpx = torch.zeros(x.shape[:-1] + (1,)).to(x)
    y, logpy = f(x, logp=logpx)

    jac = jacobian(x, y)
    jac = jac[..., :-1, :-1]
    true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    print("Logdetjac error", torch.norm(true_logdetjac + logpy, p=float("inf")))

    x_recon, logpx_recon = f.inverse(y, logp=logpy)
    print("z recon error", torch.norm(x - x_recon, p=float("inf")))
    print("logp recon error", torch.norm(logpx - logpx_recon, p=float("inf")))
    print()

    print("--- Testing logits to probs ---")

    f = LogitsToProbabilities()
    x = torch.randn(2, 10, requires_grad=True)
    x = torch.softmax(x, dim=-1)
    logpx = torch.zeros(x.shape[:-1] + (1,)).to(x)
    y, logpy = f(x, logp=logpx)

    jac = jacobian(x, y)
    jac = jac[..., :-1, :-1]
    true_logdetjac = torch.log(torch.abs(torch.det(jac))).reshape(-1, 1)
    print("Logdetjac error", torch.norm(true_logdetjac + logpy, p=float("inf")))

    x_recon, logpx_recon = f.inverse(y, logp=logpy)
    print("z recon error", torch.norm(x - x_recon, p=float("inf")))
    print("logp recon error", torch.norm(logpx - logpx_recon, p=float("inf")))
