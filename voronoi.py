"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer

import sys
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


DEBUGGING = False


def batch_dot(x, y):
    return torch.sum(x * y, dim=-1)


def batch_jacobian(outputs, inputs):
    assert inputs.shape == outputs.shape
    dim = inputs.shape[-1]
    outputs = outputs.reshape(-1, dim)

    jac = []
    for i in range(dim):
        jac.append(
            torch.autograd.grad(outputs[:, i].sum(), inputs, create_graph=True)[0]
        )
    jac = torch.stack(jac, dim=-2)
    return jac


def softsign_forward(x):
    return x / (1 + x)


def softsign_inverse(x):
    return x / (1 - x)


class VoronoiTransform(nn.Module):
    def __init__(
        self,
        num_discrete_variables,
        num_classes,
        embedding_dim,
        share_embeddings=False,
        learn_box_constraints=True,
    ):
        super().__init__()
        self.num_discrete_variables = num_discrete_variables
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.share_embeddings = share_embeddings

        n_unique_embeddings = 1 if share_embeddings else num_discrete_variables
        self._anchor_pts = nn.Parameter(
            torch.randn(n_unique_embeddings, num_classes, embedding_dim)
            / math.sqrt(embedding_dim)
        )
        self.log_scale = nn.Parameter(
            torch.log(
                torch.ones(n_unique_embeddings, num_classes)
                / math.sqrt(num_classes)
                * math.sqrt(embedding_dim)
            )
        )
        if learn_box_constraints:
            self._box_constraints = nn.Parameter(
                torch.randn(n_unique_embeddings, embedding_dim, 2)
            )
        else:
            self._box_constraints = None
        self.num_constraints = num_classes + 2 * embedding_dim

        # lambs = cp.Variable(1)
        # g = cp.Parameter(self.num_constraints)
        # h = cp.Parameter(self.num_constraints)
        # objective = cp.Maximize(cp.sum(lambs))
        # constraints = [cp.multiply(lambs, g) <= h]
        # self.prob = cp.Problem(objective, constraints)
        # self.cvxpylayer = CvxpyLayer(self.prob, parameters=[g, h], variables=[lambs])

    def extra_repr(self):
        return (
            "num_discrete_variables={num_discrete_variables}"
            ", num_classes={num_classes}"
            ", embedding_dim={embedding_dim}"
            ", share_embeddings={share_embeddings}"
        ).format(**self.__dict__)

    @property
    def anchor_pts(self):
        """Anchor pooints have values within the box constraints and have shape (N, K, D).
        """
        pts = F.softsign(self._anchor_pts)
        N, D = self.num_discrete_variables, self.embedding_dim
        pts = (pts + 1) / 2
        max_val = self.box_constraints_max.reshape(N, 1, D)
        min_val = self.box_constraints_min.reshape(N, 1, D)
        pts = pts * (max_val - min_val) + min_val
        return pts

    @property
    def box_constraints_max(self):
        """Box constraints have values greater than 1 and have shape (N, D).
        """
        N, D = self.num_discrete_variables, self.embedding_dim
        if self._box_constraints is None:
            return self._anchor_pts.new_ones(N, D)
        return (
            F.softplus(self._box_constraints[..., 0])
            .add(1.0)
            .reshape(-1, D)
            .expand(N, D)
        )

    @property
    def box_constraints_min(self):
        N, D = self.num_discrete_variables, self.embedding_dim
        if self._box_constraints is None:
            return -self._anchor_pts.new_ones(N, D)
        return (
            F.softplus(self._box_constraints[..., 1])
            .add(1.0)
            .mul(-1)
            .reshape(-1, D)
            .expand(N, D)
        )

    def find_nearest(self, x):
        """Assume x is (B, N, D). Returns a mask of shape (B, N, K)."""
        B, N, D = x.shape
        K = self.anchor_pts.shape[1]
        x = x.reshape(B, N, 1, D)
        anchor_pts = self.anchor_pts.reshape(1, -1, K, D)
        dist = torch.linalg.norm(x - anchor_pts, dim=3, keepdim=False)  # (B, N, K)
        nearest = torch.argmin(dist, 2)
        mask = F.one_hot(nearest, K).bool().reshape(B, N, K)
        return mask

    def _solve_cvxprob(self, x, mask):
        """Uses cvxpylayers to solve for lamb."""
        B, N, K, D = (
            x.shape[0],
            self.num_discrete_variables,
            self.num_classes,
            self.embedding_dim,
        )

        # Transform everything to be expandable to (B, N, K, D)
        x = x.reshape(B, N, 1, D)
        points = self.anchor_pts.reshape(1, -1, K, D).to(x)
        x_k = torch.masked_select(points, mask.reshape(B, N, K, 1)).reshape(B, N, 1, D)

        # Linear constraints for the Voronoi cell of x_k, for each discrete variable.
        A = 2 * (points - x_k)  # (B, N, K, D)
        b = batch_dot(points, points) - batch_dot(x_k, x_k)  # (B, N, K)

        # Remove the k-th constraint.
        A = A[~mask.reshape(B, N, K, 1).expand(B, N, K, D)].reshape(B, N, K - 1, D)
        b = b[~mask.reshape(B, N, K)].reshape(B, N, K - 1)

        # Add in the box constraints.
        # NOTE(rtqichen): We add it here so we can access the constraints within A later on,
        #   but if memory is an issue, we can move this to g and h.
        A = torch.cat(
            [
                A,
                torch.eye(D).reshape(1, 1, D, D).expand(B, N, D, D).to(A),
                -torch.eye(D).reshape(1, 1, D, D).expand(B, N, D, D).to(A),
            ],
            dim=2,
        )
        b = torch.cat(
            [
                b,
                self.box_constraints_max.reshape(1, N, D).expand(B, N, D).to(b),
                -self.box_constraints_min.reshape(1, N, D).expand(B, N, D).to(b),
            ],
            dim=2,
        )

        with torch.no_grad():
            del_x = (x - x_k) / (
                torch.linalg.norm(x - x_k, dim=-1, keepdim=True) + 1e-6
            )

            # Transform problem to in terms of lamb.
            g = batch_dot(A, del_x)  # (B, N, K - 1 + 2D)
            h = b - batch_dot(A, x_k)  # (B, N, K - 1 + 2D)

            # Add in the constraints: 0 <= lamb.
            g = torch.cat(
                [g, torch.Tensor([-1.0]).to(g).reshape(1, 1, 1).expand(B, N, 1)], dim=-1
            )
            h = torch.cat([h, torch.zeros(B, N, 1).to(h)], dim=-1)

            # We solve for B * N independent problems.
            g = g.reshape(B * N, self.num_constraints)
            h = h.reshape(B * N, self.num_constraints)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.cvxpylayer(
                    g,
                    h,
                    solver_args={
                        "raise_on_error": True,
                        "solve_method": "SCS",
                        "max_iters": 2000,
                        "normalize": False,
                        "eps": 1e-3,
                    },
                )

            # Re-express lamb in closed form.
            dual_vals = torch.stack(
                list(map(torch.as_tensor, self.cvxpylayer.info["ys"]))
            )
            # Remove the non-negativity constraint.
            dual_vals = dual_vals[..., :-1].reshape(B * N, self.num_constraints - 1)

            # Take the constraint with the largest dual in magnitude.
            # This is the boundary of the Voronoi cell in the direction of delta_x.
            active_cons = torch.argmax(torch.abs(dual_vals), 1, keepdim=True)
            one_hot = torch.zeros(dual_vals.shape, device=x.dual_vals)
            one_hot.scatter_(1, active_cons, 1)
            active_cons = one_hot.reshape(B, N, self.num_constraints - 1, 1).bool()

        return A, b, x_k.reshape(B, N, D), active_cons

    def _solve_cvxprob_fast(self, x, x_k, mask):
        """An exact algorithm that does not require invoking cvxpy."""
        B, N, K, D = (
            x.shape[0],
            self.num_discrete_variables,
            self.num_classes,
            self.embedding_dim,
        )

        points = self.anchor_pts.reshape(1, -1, K, D).to(x)

        # Linear constraints for the Voronoi cell of x_k, for each discrete variable.
        A = 2 * (points - x_k)  # (B, N, K, D)
        b = batch_dot(points, points) - batch_dot(x_k, x_k)  # (B, N, K)

        # Remove the k-th constraint.
        A = A[~mask.reshape(B, N, K, 1).expand(B, N, K, D)].reshape(B, N, K - 1, D)
        b = b[~mask.reshape(B, N, K)].reshape(B, N, K - 1)

        # Add in the box constraints.
        # NOTE(rtqichen): We add it here so we can access the constraints within A later on,
        #   but if memory is an issue, we can move this to g and h.
        A = torch.cat(
            [
                A,
                torch.eye(D).reshape(1, 1, D, D).expand(B, N, D, D).to(A),
                -torch.eye(D).reshape(1, 1, D, D).expand(B, N, D, D).to(A),
            ],
            dim=2,
        )
        b = torch.cat(
            [
                b,
                self.box_constraints_max.reshape(1, N, D).expand(B, N, D).to(b),
                -self.box_constraints_min.reshape(1, N, D).expand(B, N, D).to(b),
            ],
            dim=2,
        )

        with torch.no_grad():
            del_x = (x - x_k) / (
                torch.linalg.norm(x - x_k, dim=-1, keepdim=True) + 1e-6
            )

            # Transform problem to in terms of lamb.
            g = batch_dot(A, del_x)  # (B, N, num_constraints - 1)
            h = b - batch_dot(A, x_k)  # (B, N, num_constraints - 1)

            # This contains the values of lambda that cross all the other boundaries.
            lambs = h / g

            # The optimal lambda is the one that is the smallest out of all positive ones.
            lambs = torch.where(lambs > 0, lambs, torch.ones_like(lambs) * float("inf"))
            active_cons = torch.argmin(lambs, 2, keepdim=True)
            active_cons = (
                F.one_hot(active_cons, self.num_constraints - 1)
                .reshape(B, N, self.num_constraints - 1, 1)
                .bool()
            )

        b = b.reshape(B, N, self.num_constraints - 1, 1)
        a_ = A[active_cons.expand_as(A)].reshape(B, N, D)
        b_ = b[active_cons].reshape(B, N)

        return a_, b_

    def _solve_cvxprob_efficient(self, x, x_k, mask):
        """A memory efficient algorithm equivalent to the above."""
        B, N, K, D = (
            x.shape[0],
            self.num_discrete_variables,
            self.num_classes,
            self.embedding_dim,
        )

        x = x.reshape(B, N, 1, D)
        x_k = x_k.reshape(B, N, 1, D)
        x_i = self.anchor_pts.reshape(1, -1, K, D).to(x)

        all_a = torch.zeros(B, N, D).to(x)
        all_b = torch.zeros(B, N).to(x)

        with torch.no_grad():
            del_x = (x - x_k) / (torch.linalg.norm(x - x_k, dim=-1, keepdim=True))

        # Compute lamb_i = (b_i - 2<x_i,x_k> + 2<x_k, x_k>) / (<x_i, del_x> - <x_k, del_x>)
        # All quantities here are (B, N, K).
        xk_xk = batch_dot(x_k, x_k)
        b = batch_dot(x_i, x_i) - xk_xk  # (B, N, K)
        a = 2 * (x_i - x_k)
        numerator = b - batch_dot(a, x_k)
        denominator = batch_dot(a, del_x)
        lambs = numerator / denominator

        # Set the case where x_i == x_k to lamb=0.
        lambs = torch.where(denominator == 0, torch.zeros_like(lambs), lambs)

        # Handle the bounded case.
        is_bounded = torch.any(lambs > 0, dim=2)
        vor_lambs = torch.where(lambs > 0, lambs, torch.ones_like(lambs) * float("inf"))
        active_cons = torch.argmin(vor_lambs, 2)
        active_cons = F.one_hot(active_cons, K).bool()
        b_ = b[active_cons].reshape(B, N)
        a_ = a[active_cons].reshape(B, N, D)

        # Update vectors for bounded cases.
        all_a = torch.where(is_bounded.reshape(B, N, 1).expand(B, N, D), a_, all_a)
        all_b = torch.where(is_bounded, b_, all_b)

        # For the unbounded case, we use box constraints.

        # Compute lambs for the box constraints, of size (B, N, 2D).
        xk_rep = torch.cat([x_k.reshape(B, N, D), -x_k.reshape(B, N, D)], dim=2)
        box_b = torch.cat(
            [
                self.box_constraints_max.reshape(1, N, D).expand(B, N, D).to(b),
                -self.box_constraints_min.reshape(1, N, D).expand(B, N, D).to(b),
            ],
            dim=2,
        )
        numerator = box_b - xk_rep
        denominator = torch.cat(
            [del_x.reshape(B, N, D), -del_x.reshape(B, N, D)], dim=2
        )
        lambs = numerator / denominator

        box_lambs = torch.where(
            lambs >= 0, lambs, torch.ones_like(lambs) * float("inf")
        )
        active_cons = torch.argmin(box_lambs, 2)
        a_ = F.one_hot(active_cons % D, D) * (active_cons < D).to(lambs.dtype).sub(
            0.5
        ).mul(2).reshape(B, N, 1).expand(B, N, D)
        b_ = torch.gather(box_b, dim=2, index=active_cons.reshape(B, N, 1)).reshape(
            B, N
        )

        # Update vectors for unbounded cases.
        # Also update vectors if the box constraint lambda is smaller than the previous one.
        update = torch.logical_or(
            ~is_bounded,
            torch.min(box_lambs, dim=2).values < torch.min(vor_lambs, dim=2).values,
        )

        all_a = torch.where(update.reshape(B, N, 1).expand(B, N, D), a_, all_a)
        all_b = torch.where(update, b_, all_b)

        return all_a, all_b

    def map_onto_cell(self, x, mask, return_all=False):
        """
        Inputs:
            x: (batch_size, num_discrete_variables, embedding_dim)
            mask: (batch_size, num_discrete_variables, num_classes) A batch of one-hot vectors
        Returns:
            f(z) and logdet df/dz
            where f maps from R^d onto a Voronoi cell
        """
        B, N, K, D = (
            x.shape[0],
            self.num_discrete_variables,
            self.num_classes,
            self.embedding_dim,
        )

        # Transform everything to be expandable to (B, N, K, D)
        x = x.reshape(B, N, 1, D)
        points = self.anchor_pts.reshape(1, -1, K, D).to(x)
        x_k = torch.masked_select(points, mask.reshape(B, N, K, 1)).reshape(B, N, 1, D)

        a_, b_ = self._solve_cvxprob_efficient(x, x_k, mask)

        with torch.enable_grad():
            x = x.reshape(B, N, D).requires_grad_(True)
            x_k = x_k.reshape(B, N, D)

            Delta = torch.linalg.norm(x - x_k, dim=-1, keepdim=True)
            del_x = (x - x_k) / Delta

            lamb = (b_ - batch_dot(a_, x_k)) / batch_dot(a_, del_x)
            lamb = lamb.reshape(B, N, 1)

            x_lamb = x_k + lamb * del_x
            log_scale = torch.masked_select(
                self.log_scale.reshape(1, -1, K).to(mask.device), mask.reshape(B, -1, K)
            ).reshape(B, -1, 1)

            Delta_star = torch.linalg.norm(x_lamb - x_k, dim=-1, keepdim=True)

            Delta_tilde = Delta / Delta_star
            alpha = softsign_forward(Delta_tilde * torch.exp(log_scale))
            z = x_k + alpha * (x_lamb - x_k)

            dalpha = grad(alpha.sum(), Delta_tilde, create_graph=True)[0]
            logdet = self.compute_logdet(x_k, lamb, alpha, dalpha, Delta, del_x)

            # -- debugging --
            if DEBUGGING:
                true_logdet = torch.slogdet(batch_jacobian(z, x))[1]
                print("Logdet error:", (true_logdet - logdet).pow(2).mean().item())
                logdet = true_logdet
            # -- end --

            z = torch.where(Delta < 1e-6, x, z)
            logdet = torch.where(
                Delta.reshape(B, N) < 1e-6, torch.zeros_like(logdet), logdet
            )

            # if not torch.isfinite(z).all() or not torch.isfinite(logdet).all():
            #     ipdb.set_trace()

        if return_all:
            return z, x_k, x_lamb, logdet
        else:
            return z, logdet

    def map_outside_cell(self, z, return_all=False):
        """
        Inputs:
            z: (batch_size, num_discrete_variables, embedding_dim)
        Returns:
            f(z) and logdet df/dz
            where f maps from a Voronoi cell onto R^d
        """
        B, N, K, D = (
            z.shape[0],
            self.num_discrete_variables,
            self.num_classes,
            self.embedding_dim,
        )

        mask = self.find_nearest(z)

        # Transform everything to be expandable to (B, N, K, D)
        z = z.reshape(B, N, 1, D)
        points = self.anchor_pts.reshape(1, -1, K, D).to(z)
        x_k = torch.masked_select(points, mask.reshape(B, N, K, 1)).reshape(B, N, 1, D)

        a_, b_ = self._solve_cvxprob_efficient(z, x_k, mask)

        with torch.enable_grad():
            z = z.reshape(B, N, D).requires_grad_(True)
            x_k = x_k.reshape(B, N, D)

            Delta_z = torch.linalg.norm(z - x_k, dim=-1, keepdim=True)
            del_x = (z - x_k) / Delta_z

            lamb = (b_ - batch_dot(a_, x_k)) / batch_dot(a_, del_x)
            lamb = lamb.reshape(B, N, 1)
            x_lamb = x_k + lamb * del_x
            log_scale = torch.masked_select(
                self.log_scale.reshape(1, -1, K).to(mask.device), mask.reshape(B, -1, K)
            ).reshape(B, -1, 1)

            alpha = ComputeAlpha.apply(z, x_lamb, x_k)

            if torch.isnan(alpha).any() or (alpha < 0).any() or (alpha > 1).any():
                print(f"Alphas are in ({alpha.min()}, {alpha.max()})", file=sys.stderr)
                # ipdb.set_trace()

            alpha = torch.clamp(alpha, min=1e-6, max=1.0 - 1e-6)

            Delta_ratio = softsign_inverse(alpha) / torch.exp(log_scale)
            Delta_star = torch.linalg.norm(x_lamb - x_k, dim=-1, keepdim=True)
            Delta_x = Delta_ratio * Delta_star
            x = del_x * Delta_x + x_k

            dalpha = torch.reciprocal(
                grad(Delta_ratio.sum(), alpha, create_graph=True)[0].clamp(min=1e-12)
            )
            logdet = -self.compute_logdet(x_k, lamb, alpha, dalpha, Delta_x, del_x)

            # -- debugging --
            if DEBUGGING:
                true_logdet = torch.slogdet(batch_jacobian(x, z))[1]
                print("Logdet error:", (true_logdet - logdet).pow(2).mean().item())
                logdet = true_logdet
            # -- end --

            x = torch.where(Delta_z < 1e-6, z, x)
            logdet = torch.where(
                Delta_z.reshape(B, N) < 1e-6, torch.zeros_like(logdet), logdet
            )

        if return_all:
            return x, x_k, x_lamb, logdet
        else:
            return x, logdet

    def compute_logdet(self, x_k, lamb, alpha, dalpha, Delta, del_x):
        """
        Inputs:
            x_k: (batch_size, num_discrete_variables, embedding_dim)
            lamb: (batch_size, num_discrete_variables, 1)
            alpha: (batch_size, num_discrete_variables, 1)
            dalpha: (batch_size, num_discrete_variables, 1)
            Delta: (batch_size, num_discrete_variables, 1)
            del_x: (batch_size, num_discrete_variables, embedding_dim)
        Returns:
            logdet: (batch_size, num_discrete_variables)
        """
        B, N, D = x_k.shape

        dlamb = grad(lamb.sum(), del_x, create_graph=self.training, retain_graph=True)[
            0
        ]

        dotprod_delx = torch.sum(del_x * del_x, dim=-1, keepdim=True)
        dotprod_dlamb_delx = torch.sum(dlamb * del_x, dim=-1, keepdim=True)

        c = torch.div(alpha * lamb, Delta)
        u1 = torch.div(alpha, Delta).sub(dalpha * dotprod_delx).mul(del_x)
        v1 = dlamb
        u2 = (
            dalpha
            - c
            - alpha / Delta * dotprod_dlamb_delx
            + lamb * dalpha * (dotprod_delx - 1.0)
            + dalpha * dotprod_delx * dotprod_dlamb_delx
        ) * del_x
        v2 = del_x

        w11 = batch_dot(v1, u1) / c.reshape(B, N)  # (B, N)
        w22 = batch_dot(v2, u2) / c.reshape(B, N)  # (B, N)
        w12 = batch_dot(v1, u2) / c.reshape(B, N)  # (B, N)
        w21 = batch_dot(v2, u1) / c.reshape(B, N)  # (B, N)

        logdet = (
            safe_log(torch.abs(1 + w11))
            + safe_log(torch.abs(1 + w22 - w12 * w21 / torch.clamp(1 + w11, min=1e-22)))
            + D * safe_log(c.reshape(B, N))
        )

        return logdet

    @torch.autograd.no_grad()
    def find_nearest(self, x):
        """
        Inputs:
            x: (batch_size, num_discrete_variables, embedding_dim)
            anchor_pts: (num_discrete_variables, num_classes, embedding_dim)
        """
        B, N, D = x.shape
        K = self.anchor_pts.shape[1]
        x = x.reshape(B, N, 1, D)
        anchor_pts = self.anchor_pts.reshape(1, N, K, D)
        dist = torch.linalg.norm(x - anchor_pts, dim=3, keepdim=False)  # (B, N, K)
        nearest = torch.argmin(dist, 2, keepdim=True)
        one_hot = torch.zeros(dist.shape, device=dist.device)
        one_hot.scatter_(dim=2, index=nearest, value=1)
        mask = one_hot.bool()
        return mask


def safe_log(x):
    return torch.log(x.clamp(min=1e-22))


class ComputeAlpha(torch.autograd.Function):
    """Computes alpha and handle the case of 0/0."""

    @staticmethod
    def forward(ctx, z, x_lamb, x_k):
        alpha = torch.nanmean((z - x_k) / (x_lamb - x_k), dim=-1, keepdim=True)
        ctx.save_for_backward(z, x_lamb, x_k)
        ctx.dim = z.shape[-1]
        return alpha

    @staticmethod
    def backward(ctx, grad_output):
        z, x_lamb, x_k = ctx.saved_tensors
        denom = x_lamb - x_k

        grad_output = grad_output / ctx.dim

        grad_z = grad_output * torch.reciprocal(denom)
        grad_z[denom == 0] = 0

        grad_x_lamb = grad_output * (x_k - z) / denom ** 2
        grad_x_lamb[denom == 0] = 0

        grad_x_k = grad_output * (z - x_lamb) / denom ** 2
        grad_x_k[denom == 0] = 0

        return grad_z, grad_x_lamb, grad_x_k
