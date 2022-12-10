"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers as layers
import layers.base as base_layers

import voronoi

ACT_FNS = {
    "softplus": lambda b: nn.Softplus(),
    "elu": lambda b: nn.ELU(inplace=b),
    "swish": lambda b: base_layers.Swish(),
    "lcube": lambda b: base_layers.LipschitzCube(),
    "identity": lambda b: base_layers.Identity(),
    "relu": lambda b: nn.ReLU(inplace=b),
}


class MultiscaleFlow(nn.Module):
    """ Creates a stack of flow blocks with squeeze / factor out.
    Main arg:
        block_fn: Function that takes an index i, a 3D input shape (c, h, w), and a boolean (fc), returns an invertible block.
    """

    def __init__(
        self,
        input_size,
        block_fn,
        n_blocks=[16, 16],
        factor_out=True,
        init_layer=None,
        actnorm=False,
        fc_end=False,
        voronoi_tessellation=False,
        n_mixtures=10,
        cond_embed_dim_factor=4,
        lazy_init=False,
        skip_transform=False,
    ):
        super(MultiscaleFlow, self).__init__()
        self.n_scale = len(n_blocks)
        self.n_blocks = n_blocks
        self.factor_out = factor_out
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_end = fc_end
        self.voronoi_tessellation = voronoi_tessellation
        self.n_mixtures = n_mixtures
        self.cond_embed_dim_factor = cond_embed_dim_factor
        self.lazy_init = lazy_init
        self.skip_transform = skip_transform

        self.transforms = self._build_net(input_size, block_fn)
        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

    def _build_net(self, input_size, block_fn):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedInvBlocks(
                    block_fn=block_fn,
                    initial_size=(c, h, w),
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    actnorm=self.actnorm,
                    fc_end=self.fc_end,
                    voronoi_tessellation=self.voronoi_tessellation if i > 0 else False,
                    n_mixtures=self.n_mixtures,
                    cond_embed_dim_factor=self.cond_embed_dim_factor,
                    lazy_init=self.lazy_init,
                    skip_transform=self.skip_transform,
                )
            )
            c, h, w = c * 2 if self.factor_out else c * 4, h // 2, w // 2
        return nn.ModuleList(transforms)

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4 ** k, h // 2 ** k, w // 2 ** k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, inverse=False):
        if inverse:
            return self.inverse(x, logpx)
        out = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                x, logpx = self.transforms[idx].forward(x, logp=logpx)
            else:
                x = self.transforms[idx].forward(x)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        return output

    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i : i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logp=logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logp=logpz)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].inverse(z)
                else:
                    z, logpz = self.transforms[idx].inverse(z, logp=logpz)
            return z if logpz is None else (z, logpz)


class StackedInvBlocks(layers.SequentialFlow):
    def __init__(
        self,
        block_fn,
        initial_size,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        actnorm=False,
        fc_end=False,
        voronoi_tessellation=False,
        n_mixtures=10,
        cond_embed_dim_factor=4,
        lazy_init=False,
        skip_transform=False,
    ):

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        if init_layer is not None:
            chain.append(init_layer)
            chain.append(_actnorm(initial_size, fc=False))

        chain.append(
            VoronoiMixtureFlow(
                initial_size=initial_size,
                block_fn=block_fn,
                n_blocks=n_blocks,
                squeeze=squeeze,
                fc_end=fc_end,
                actnorm=actnorm,
                voronoi_tessellation=voronoi_tessellation,
                n_mixtures=n_mixtures,
                cond_embed_dim_factor=cond_embed_dim_factor,
                lazy_init=lazy_init,
                skip_transform=skip_transform,
            )
        )

        super(StackedInvBlocks, self).__init__(chain)


class FCNet(nn.Module):
    def __init__(
        self,
        input_shape,
        idim,
        lipschitz_layer,
        nhidden,
        coeff,
        domains,
        codomains,
        n_iterations,
        activation_fn,
        preact,
        dropout,
        sn_atol,
        sn_rtol,
        learn_p,
        div_in=1,
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        dim = c * h * w
        nnet = []
        last_dim = dim // div_in
        if preact:
            nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.0)) for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim)
                if lipschitz_layer == nn.Linear
                else lipschitz_layer(
                    last_dim,
                    idim,
                    coeff=coeff,
                    n_iterations=n_iterations,
                    domain=domains[i],
                    codomain=codomains[i],
                    atol=sn_atol,
                    rtol=sn_rtol,
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout:
            nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim)
            if lipschitz_layer == nn.Linear
            else lipschitz_layer(
                last_dim,
                dim,
                coeff=coeff,
                n_iterations=n_iterations,
                domain=domains[-1],
                codomain=codomains[-1],
                atol=sn_atol,
                rtol=sn_rtol,
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.nnet(x)
        return y.view(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):
    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, *, logp=None, **kwargs):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logp is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logp = self.fc_module(x, logp=logp)
            return y.view(*shape), logp

    def inverse(self, y, *, logp=None, **kwargs):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logp is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logp = self.fc_module.inverse(y, logp=logp)
            return x.view(*shape), logp


class VoronoiTessellation(nn.Module):
    def __init__(self, num_anchor_pts, dim, lazy_init=False, skip_transform=False):
        super().__init__()
        self.num_anchor_pts = num_anchor_pts
        self.dim = dim

        self.alpha = 0.01
        self.actnorm1 = layers.ActNorm1d(dim)
        self.actnorm2 = layers.ActNorm1d(dim)
        self.logit_transform = layers.LogitTransform(alpha=self.alpha)
        self.shift_coeff = nn.Parameter(torch.randn(dim))
        self.vorproj = voronoi.VoronoiTransform(1, num_anchor_pts, dim)
        self.mixture_logits = nn.Parameter(torch.zeros(num_anchor_pts))
        self.register_buffer("lazy_init", torch.tensor(1 if lazy_init else 0))
        self.skip_transform = skip_transform

        # A hack for storing the discrete values.
        self.mask = None

    @property
    def boundary_eps(self):
        return self.alpha / (1 - 2 * self.alpha)

    def forward(self, x, *, logp=None, **kwargs):
        del kwargs

        input_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        B, D = x.shape

        return_logp = logp is not None
        if logp is None:
            logp = torch.zeros(x.shape[0], 1, device=x.device)

        if not self.skip_transform:
            x, logp = self.actnorm1(x, logp=logp)

            # Transform with logit transform to bounded domain [-eps, 1 + eps]
            x, logp = self.logit_transform.inverse(x, logp=logp)

            # Transform from [-eps, 1 + eps] to [0, 1]
            min_val = -self.boundary_eps
            max_val = 1 + self.boundary_eps
            x = (x - min_val) / (max_val - min_val)
            logdet = (
                -torch.log(torch.tensor(max_val - min_val, device=x.device))
                .reshape(1, 1)
                .expand(B, D)
                .sum(1, keepdim=True)
            )
            logp = logp - logdet

            # Transform from [0, 1] into voronoi transform box boundaries.
            max_val = self.vorproj.box_constraints_max.reshape(1, D).to(x.dtype)
            min_val = self.vorproj.box_constraints_min.reshape(1, D).to(x.dtype)
            x = x * (max_val - min_val) + min_val
            logdet = (
                torch.log(max_val - min_val)
                .reshape(1, D)
                .expand(B, D)
                .sum(1, keepdim=True)
            )
            logp = logp - logdet

        # Lazily initialize the anchor points.
        if self.lazy_init:
            if B >= self.num_anchor_pts:
                x_np = x.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=self.vorproj.num_classes)
                kmeans.fit(x_np)
                pts = torch.as_tensor(kmeans.cluster_centers_).to(x)
                max_val = self.vorproj.box_constraints_max.reshape(1, -1)
                min_val = self.vorproj.box_constraints_min.reshape(1, -1)
                pts = (pts - min_val) / (max_val - min_val)
                pts = pts * 2 - 1
                pts = pts / (1 - torch.abs(pts))
                self.vorproj._anchor_pts.data.copy_(pts[None])
            self.lazy_init.data.fill_(0)

        # Get closest anchor point.
        x = x.reshape(B, 1, D)
        mask = self.vorproj.find_nearest(x)

        # A hack for storing the discrete values.
        self.mask = mask

        if not self.skip_transform:
            # Add perturbations to be closer to anchor points.
            K = mask.shape[-1]
            x = x.reshape(B, 1, D)
            points = self.vorproj.anchor_pts.reshape(1, K, D).to(x)
            x_k = torch.masked_select(points, mask.reshape(B, K, 1)).reshape(B, 1, D)
            shift_coeff = torch.sigmoid(self.shift_coeff) * 0.98 + 0.01
            x = x + shift_coeff * (x_k - x)
            logdet = (
                torch.log(1 - shift_coeff)
                .reshape(1, D)
                .expand(B, D)
                .sum(1, keepdim=True)
            )
            logp = logp - logdet

            # Transform with voronoi projection to outside the cell.
            x, logdet = self.vorproj.map_outside_cell(x)
            logp = logp - logdet

            x = x.reshape(B, D)

            # Shift all points to the origin based on their anchor point.
            pts = self.vorproj.anchor_pts.unsqueeze(0).expand(B, -1, -1, -1)
            _mask = mask.unsqueeze(-1).expand_as(pts)
            centers = pts[_mask].reshape(B, D)
            x = x - centers

            x, logp = self.actnorm2(x, logp=logp)
        else:
            x = x.reshape(B, D)

        # Add log probability of the mixture component.
        mixture_logprobs = F.log_softmax(self.mixture_logits, dim=0)
        logp_k = torch.masked_select(mixture_logprobs, mask).reshape(-1, 1)

        # Kind of a hack for including logp_k in the objective.
        # Negative since we ultimately take the negative of this quantity.
        logp = logp - logp_k

        x = x.reshape(input_shape)

        if return_logp:
            return x, logp, mask
        else:
            return x, None, mask

    def sample_mask(self, z):
        B = z.shape[0]

        # A hack for storing the discrete values.
        if self.mask is not None:
            return self.mask

        logits = F.log_softmax(self.mixture_logits, dim=0)
        ks = torch.distributions.Categorical(logits=logits).sample((B,))
        mask = F.one_hot(ks, self.num_anchor_pts).reshape(B, 1, -1).bool()
        return mask

    def inverse(self, z, mask, *, logp=None):
        assert (
            logp is None
        ), "Log probability computation during inverse is not implemented."

        input_shape = z.shape
        z = z.reshape(z.shape[0], -1)
        B, D = z.shape

        if not self.skip_transform:
            z = self.actnorm2.inverse(z)

            # Invert: Shift all points to the origin based on their anchor point.
            pts = self.vorproj.anchor_pts.unsqueeze(0).expand(B, -1, -1, -1)
            _mask = mask.unsqueeze(-1).expand_as(pts)
            centers = pts[_mask].reshape(B, D)
            z = z + centers

            z = z.reshape(B, 1, D)
            z, _ = self.vorproj.map_onto_cell(z, mask)

            # Invert: add perturbations to be closer to anchor points.
            K = mask.shape[-1]
            points = self.vorproj.anchor_pts.reshape(1, K, D).to(z)
            x_k = torch.masked_select(points, mask.reshape(B, K, 1)).reshape(B, 1, D)
            shift_coeff = torch.sigmoid(self.shift_coeff) * 0.98 + 0.01
            z = (z - shift_coeff * x_k) / (1 - shift_coeff)

            z = z.reshape(B, D)
            z = self._invert_from_voronoi_coordinates(z)

        return z.reshape(*input_shape)

    def _invert_from_voronoi_coordinates(self, z):
        D = z.shape[-1]

        # Invert: transform from [0, 1] into voronoi transform box boundaries.
        max_val = self.vorproj.box_constraints_max.reshape(1, D).to(z.dtype)
        min_val = self.vorproj.box_constraints_min.reshape(1, D).to(z.dtype)
        z = (z - min_val) / (max_val - min_val)

        # Invert: transform from [-eps, 1 + eps] to [0, 1].
        min_val = -self.boundary_eps
        max_val = 1 + self.boundary_eps
        z = z * (max_val - min_val) + min_val

        # Invert: transform with logit transform to bounded domain [-eps, 1 + eps].
        z = self.logit_transform(z)

        z = self.actnorm1.inverse(z)
        return z

    @torch.no_grad()
    def tessellation_edges(self, length=100):

        device = self.vorproj.anchor_pts.device

        points = self.vorproj.anchor_pts.detach().cpu().squeeze(0).numpy()
        N, D = points.shape
        assert D == 2

        from scipy.spatial import Voronoi
        vor = Voronoi(points)

        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)

        line_segments = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                line_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[i] + direction * ptp_bound.max()

                line_segments.append([vor.vertices[i], far_point])

        full_segments = []
        for (x, y) in line_segments:
            x, y = torch.as_tensor(x), torch.as_tensor(y)
            interp = torch.linspace(0, 1, length).reshape(-1, 1).expand(-1, 2)
            full_segments.append(x + interp * (y - x))

        full_segments = torch.stack(full_segments, dim=0).to(device)
        M, L, D = full_segments.shape
        z = full_segments.reshape(M * L, D)
        z = self._invert_from_voronoi_coordinates(z)

        full_segments = z.reshape(M, L, D)
        return full_segments


class VoronoiMixtureFlow(nn.Module):
    def __init__(
        self,
        *,
        initial_size,
        block_fn,
        n_blocks,
        squeeze,
        fc_end,
        actnorm,
        voronoi_tessellation,
        n_mixtures,
        cond_embed_dim_factor=4,
        lazy_init=False,
        skip_transform=False,
    ):
        super().__init__()

        self.squeeze = squeeze

        if voronoi_tessellation:
            self.cond_embed_dim = initial_size[0] * cond_embed_dim_factor
        else:
            self.cond_embed_dim = 0

        if voronoi_tessellation:
            dim = np.prod(initial_size)
            self.vor_mixture = VoronoiTessellation(
                n_mixtures, dim, lazy_init=lazy_init, skip_transform=skip_transform
            )
            if self.cond_embed_dim > 0:
                self.embed_class = nn.Embedding(n_mixtures, self.cond_embed_dim)
            else:
                self.embed_class = None
        else:
            self.vor_mixture = None
            self.embed_class = None

        use_cond_flow = voronoi_tessellation and self.cond_embed_dim > 0

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _condaffine(size, fc):
            if fc:
                dim = size[0] * size[1] * size[2]
                return FCWrapper(
                    layers.ConditionalAffine1d(
                        dim,
                        nn.Sequential(
                            nn.Linear(self.cond_embed_dim, 128),
                            nn.GELU(),
                            nn.Linear(128, 128),
                            nn.GELU(),
                            nn.Linear(128, dim * 2),
                        ),
                    )
                )
            else:
                return layers.ConditionalAffine2d(
                    size[0],
                    nn.Sequential(
                        nn.Linear(self.cond_embed_dim, 128),
                        nn.GELU(),
                        nn.Linear(128, 128),
                        nn.GELU(),
                        nn.Linear(128, size[0] * 2),
                    ),
                )

        chain = []
        if squeeze:
            if actnorm:
                chain.append(_actnorm(initial_size, fc=False))
                if use_cond_flow:
                    chain.append(_condaffine(initial_size, fc=False))
            for i in range(n_blocks):
                chain.append(
                    block_fn(
                        i,
                        initial_size,
                        fc=False,
                        cond_embed_dim=self.cond_embed_dim,
                        cond_dim=1 if use_cond_flow else None,
                    )
                )
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=False))
                    if use_cond_flow:
                        chain.append(_condaffine(initial_size, fc=False))
            chain.append(layers.SqueezeLayer(2))
        else:
            if actnorm:
                chain.append(_actnorm(initial_size, fc=False))
                if use_cond_flow:
                    chain.append(_condaffine(initial_size, fc=False))
            for i in range(n_blocks):
                chain.append(
                    block_fn(
                        i,
                        initial_size,
                        fc=False,
                        cond_embed_dim=self.cond_embed_dim,
                        cond_dim=1 if use_cond_flow else None,
                    )
                )
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=False))
                    if use_cond_flow:
                        chain.append(_condaffine(initial_size, fc=False))
            # Use one fully connected block at the end.
            if fc_end:
                chain.append(
                    FCWrapper(
                        block_fn(
                            i,
                            initial_size,
                            fc=True,
                            cond_embed_dim=self.cond_embed_dim,
                            cond_dim=1 if use_cond_flow else None,
                        )
                    )
                )
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=True))
                    if use_cond_flow:
                        chain.append(_condaffine(initial_size, fc=True))

        self.flow = layers.SequentialFlow(chain)

    def forward(self, x, logp=None, **kwargs):
        if self.vor_mixture is None:
            return self.flow(x, logp=logp, **kwargs)

        B, _, H, W = x.shape

        x, logp, mask = self.vor_mixture(x, logp=logp)

        if self.cond_embed_dim > 0:
            idx = torch.argmax(mask.to(x.dtype), dim=-1).reshape(B)
            cond_embedding = (
                self.embed_class(idx)
                .reshape(B, self.cond_embed_dim, 1, 1)
                .expand(B, -1, H, W)
            ) / np.sqrt(self.cond_embed_dim)
        else:
            cond_embedding = None

        return self.flow(x, logp=logp, cond=cond_embedding, **kwargs)

    def inverse(self, z, logp=None, **kwargs):
        if self.vor_mixture is None:
            return self.flow.inverse(z, logp=logp, **kwargs)

        B, _, H, W = z.shape

        mask = self.vor_mixture.sample_mask(z)

        if self.cond_embed_dim > 0:
            idx = torch.argmax(mask.to(z.dtype), dim=-1).reshape(B)
            H = H * 2 if self.squeeze else H
            W = W * 2 if self.squeeze else W
            cond_embedding = (
                self.embed_class(idx)
                .reshape(B, self.cond_embed_dim, 1, 1)
                .expand(B, -1, H, W)
            ) / np.sqrt(self.cond_embed_dim)
        else:
            cond_embedding = None

        z = self.flow.inverse(z, cond=cond_embedding, **kwargs)
        z = self.vor_mixture.inverse(z, mask)
        return z
