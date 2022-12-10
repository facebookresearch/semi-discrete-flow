"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argmax_utils
import layers
import layers.diffeq_layers as diffeq_layers
from model_utils import normal_logpdf, standard_normal_logpdf
from model_utils import actfns, ConditionalGaussianDistribution
import voronoi


class DeterminantalPointProcesses(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self._L_chol = nn.Parameter(
            torch.randn(num_classes, num_classes) / math.sqrt(num_classes)
        )

    @property
    def L(self):
        return torch.matmul(self._L_chol, self._L_chol.T)

    def forward(self, x):
        bsz, k = x.shape
        _x = torch.sum(F.one_hot(x, self.num_classes), dim=1)
        mask = _x.reshape(-1, self.num_classes, 1) * _x.reshape(-1, 1, self.num_classes)
        L_x = torch.masked_select(self.L[None], mask.bool()).reshape(bsz, k, k)
        loglik = torch.logdet(L_x)
        normalization = torch.logdet(self.L + torch.eye(self.num_classes).to(self.L))
        return loglik - normalization


class DiscreteSetsFlow(nn.Module):
    def __init__(
        self,
        dequantization,
        num_classes,
        embedding_dim,
        flow_type,
        num_flows,
        num_layers,
        actfn,
    ):
        super().__init__()
        self.dequantization = dequantization
        self.num_classes = num_classes

        if self.dequantization == "voronoi":
            self.embedding_dim = embedding_dim
        elif self.dequantization == "argmax":
            self.embedding_dim = int(np.ceil(np.log2(num_classes)))
        elif self.dequantization == "simplex":
            self.embedding_dim = num_classes - 1
        else:
            raise ValueError(f"Unknown dequantization {self.dequantization}")

        del embedding_dim

        self.embed_tokens = nn.Embedding(num_classes, self.embedding_dim)
        self.dequant_base = ConditionalGaussianDistribution(
            self.embedding_dim, self.embedding_dim, actfn
        )

        if self.dequantization == "voronoi":
            self.voronoi_transform = voronoi.VoronoiTransform(
                1, num_classes, self.embedding_dim, share_embeddings=True
            )
        elif self.dequantization == "argmax":
            self.softplus = layers.Softplus()
        elif self.dequantization == "simplex":
            self.logit_centershift = layers.LogitCenterShift(0.01)
            self.log_softmax = layers.InvertibleLogSoftmax()
            self.cond_logits = layers.ConditionalLogits()
        else:
            raise ValueError(f"Unknown dequantization {self.dequantization}")

        if flow_type == "cnf":

            def flow_layer(i):
                return layers.CNF(
                    DiffEqTransformer(
                        num_layers=num_layers,
                        d_in=self.embedding_dim,
                        d_out=self.embedding_dim,
                        actfn=actfn,
                    ),
                    nonself_connections=False,  # TODO: impl this.
                )

        elif flow_type == "coupling":

            def flow_layer(i):
                return layers.MaskedCouplingBlock(
                    Transformer(
                        num_layers=num_layers,
                        d_in=self.embedding_dim,
                        d_out=self.embedding_dim * 2,
                        actfn=actfn,
                    ),
                    mask_dim=2,
                    split_dim=2,
                    mask_type={0: "channel0", 1: "channel1",}[i % 2],
                )

        flows = [layers.ActNorm1d(self.embedding_dim)]
        for i in range(num_flows):
            flows.append(flow_layer(i))
            flows.append(layers.ActNorm1d(self.embedding_dim))
        self.flow = layers.SequentialFlow(flows)

    def forward(self, x):
        """x is integer-valued and of shape (batch size x seqlen)"""
        batch_size, seqlen = x.shape
        K, D = self.num_classes, self.embedding_dim
        cond = self.embed_tokens(x)
        z, logqz = self.dequant_base.sample(cond=cond)

        if self.dequantization == "voronoi":

            z = z.reshape(batch_size * seqlen, 1, D)

            # Center the flow at the Voronoi cell.
            mask = F.one_hot(x, self.num_classes).bool()
            points = self.voronoi_transform.anchor_pts.reshape(1, 1, K, D)
            x_k = torch.masked_select(points, mask.reshape(-1, 1, K, 1)).reshape(
                -1, 1, D
            )
            z = z + x_k
            # Transform into the target Voronoi cell.
            z, logdet = self.voronoi_transform.map_onto_cell(z, mask=mask)
            logdet = logdet.reshape(batch_size, seqlen).sum(1, keepdim=True)
            logqz = logqz - logdet

            z = z.reshape(batch_size, seqlen, D)

        elif self.dequantization == "argmax":

            z = z.reshape(batch_size * seqlen, 1, D)

            binary = argmax_utils.integer_to_base(
                x.reshape(batch_size * seqlen, 1), base=2, dims=self.embedding_dim
            )
            sign = binary * 2 - 1

            z, neg_logdet = self.softplus(
                z, logp=torch.zeros(batch_size * seqlen, 1, device=z.device)
            )
            logqz = logqz + neg_logdet.reshape(batch_size, seqlen).sum(1, keepdim=True)
            z = z * sign
            z = z.reshape(batch_size, seqlen, D)

        elif self.dequantization == "simplex":

            z = z.reshape(batch_size * seqlen, 1, D)
            x = x.reshape(batch_size * seqlen, 1)

            neg_logdet = torch.zeros(batch_size * seqlen, 1, device=z.device)

            z, neg_logdet = self.cond_logits(z, x, logp=neg_logdet)
            z, neg_logdet = self.log_softmax(z, logp=neg_logdet)
            z, neg_logdet = self.logit_centershift(z, logp=neg_logdet)
            z, neg_logdet = self.log_softmax.inverse(z, logp=neg_logdet)
            logqz = logqz + neg_logdet.reshape(batch_size, seqlen).sum(1, keepdim=True)

            z = z.reshape(batch_size, seqlen, D)

        else:
            assert False

        z, logqz = self.flow(z, logp=logqz)
        logpz = standard_normal_logpdf(z).reshape(z.shape[0], -1).sum(1, keepdim=True)
        loglik = logpz - logqz
        return loglik


class ConditionalGaussianDistribution(nn.Module):
    def __init__(self, d, cond_embed_dim, actfn):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(cond_embed_dim, 1024), actfns[actfn](), nn.Linear(1024, d * 2)
        )

    def _get_params(self, cond):
        out = self.net(cond)
        mean, log_std = torch.split(out, (self.d, self.d), dim=-1)
        return mean, log_std

    def forward(self, z, cond):
        mean, log_std = self._get_params(cond)
        return normal_logpdf(z, mean, log_std).sum(1, keepdim=True)

    def sample(self, cond):
        mean, log_std = self._get_params(cond)
        z = torch.randn(*cond.shape[:-1], self.d, device=cond.device)
        z = z * log_std.exp() + mean
        log_p = (
            normal_logpdf(z, mean, log_std)
            .reshape(cond.shape[0], -1)
            .sum(1, keepdim=True)
        )
        return z, log_p


class DiffEqTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_in,
        d_out,
        d_model=512,
        nhead=8,
        dim_feedforward=512,
        actfn="relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.linear1 = diffeq_layers.ConcatLinear(d_in, d_model)
        self.layers = nn.ModuleList(
            [
                DiffEqEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=actfns[actfn](),
                )
                for _ in range(num_layers)
            ]
        )
        self.linear2 = diffeq_layers.ConcatLinear(d_model, d_out)

        self.linear2._layer.weight.data.div_(d_out)
        self.linear2._layer.bias.data.zero_()

    def forward(self, t, x):
        """Assumes x is (batch size x seqlen x dim)."""

        # Convert from (B, L, D) to (L, B, D).
        x = x.transpose(0, 1)

        if self.linear1 is not None:
            x = self.linear1(t, x)

        for mod in self.layers:
            x = mod(t, x)
        x = self.linear2(t, x)

        # Convert back.
        x = x.transpose(0, 1)

        return x


class Transformer(DiffEqTransformer):
    def forward(self, x):
        t = torch.zeros(1, device=x.device)
        return super().forward(t, x)


class DiffEqEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, activation=F.relu,
    ):
        super(DiffEqEncoderLayer, self).__init__()
        self.self_attn = L2MultiheadAttention(d_model, nhead)

        self.linear1 = diffeq_layers.ConcatLinear(d_model, dim_feedforward)
        self.linear2 = diffeq_layers.ConcatLinear(dim_feedforward, d_model)

        self.norm1 = layers.ActNorm1d(d_model)
        self.norm2 = layers.ActNorm1d(d_model)

        if isinstance(activation, str):
            self.activation = actfns[activation]()
        else:
            self.activation = activation

    def forward(self, t, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(t, x))
        return x

    # self-attention block
    def _sa_block(self, x):
        x = self.self_attn(x)
        return x

    # feed forward block
    def _ff_block(self, t, x):
        x = self.linear2(t, self.activation(self.linear1(t, x)))
        return x


class L2MultiheadAttention(nn.Module):
    """ Kim et al. "The Lipschitz Constant of Self-Attention" https://arxiv.org/abs/2006.04710 """

    def __init__(self, embed_dim, num_heads):
        super(L2MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_weight.view(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.view(self.embed_dim, self.embed_dim))

    def forward(self, x, attn_mask=None, rm_nonself_grads=False):
        """
        Args:
            x: (T, N, D)
            attn_mask: (T, T) added to pre-softmax logits.
        """

        T, N, _ = x.shape

        q = k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (
            torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
            + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
            - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k)
        )
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[..., None, None]
            attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=1)  # (T, S, N, H)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(
            self.head_dim
        )
        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)

        if rm_nonself_grads:
            # Construct self-only gradient paths wrt keys and queries.
            q_detach = q.detach()
            k_detach = k.detach()
            attn_logits_keyonly = -(
                torch.einsum("tbhd,tbhd->tbh", q_detach, q_detach).unsqueeze(1)
                + torch.einsum("sbhd,sbhd->sbh", k, k).unsqueeze(0)
                - 2 * torch.einsum("tbhd,sbhd->tsbh", q_detach, k)
            ) / math.sqrt(self.head_dim)
            attn_logits_queryonly = -(
                torch.einsum("tbhd,tbhd->tbh", q, q).unsqueeze(1)
                + torch.einsum("sbhd,sbhd->sbh", k_detach, k_detach).unsqueeze(0)
                - 2 * torch.einsum("tbhd,sbhd->tsbh", q, k_detach)
            ) / math.sqrt(self.head_dim)

            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (
                attn_logits_keyonly - attn_logits_keyonly.detach()
            )
            if attn_mask is not None:
                attn_logits += attn_mask
            attn_weights = F.softmax(attn_logits, dim=1)

            # Zero out the nonself weights.
            selfonly_mask = ~(
                torch.triu(torch.ones(T, T), diagonal=1)
                + torch.tril(torch.ones(T, T), diagonal=-1)
            ).bool()
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None].to(
                attn_weights.device
            )
            # Self-only gradient path wrt values.
            PXA_vpath = torch.einsum(
                "tsbh,sbhm->tbhm", selfonly_attn_weights.detach(), XA
            )
            PXA_spath = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA.detach())

            modified_PXA = PXA_spath + (PXA_vpath - PXA_vpath.detach())
            PXA = PXA.detach() + (modified_PXA - modified_PXA.detach())

        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(
            T, N, self.embed_dim
        )
        return self.out_proj(PXAV)


class SelfonlyGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_logits):
        return attn_logits

    @staticmethod
    def backward(ctx, grads):
        # (T, T, N, H) -> (N, H, T)
        grads = torch.diagonal(grads, dim1=0, dim2=1)
        # (N, H, T, T) -> (T, T, N, H)
        grads = torch.diag_embed(grads).permute(2, 3, 0, 1)
        return grads
