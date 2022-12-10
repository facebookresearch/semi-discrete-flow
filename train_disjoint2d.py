"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=0)
from setproctitle import setproctitle

setproctitle("discrete2d")

from omegaconf import OmegaConf
import hydra

import math
from functools import partial
import logging
import os
import time
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from multiscale_flow import VoronoiTessellation
from toy_data import inf_train_gen
import layers

log = logging.getLogger(__name__)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")

        self.iter = 0

    def run(self):

        log.info(f"Work directory is {self.work_dir}")
        log.info("Running with configuration:\n" + OmegaConf.to_yaml(self.cfg))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            log.info("Found {} CUDA devices.".format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log.info(
                    "{} \t Memory: {:.2f}GB".format(
                        props.name, props.total_memory / (1024 ** 3)
                    )
                )
        else:
            log.info("WARNING: Using device {}".format(device))

        self.device = device
        self.use_gpu = device.type == "cuda"

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        self.main()

    def initialize(self):

        if self.cfg.block_type == "coupling":
            nblocks = self.cfg.coupling.nblocks
            block_fn = partial(
                layers.coupling_block_fn,
                fc=True,
                idim=self.cfg.idim,
                zero_init=self.cfg.zero_init,
                depth=self.cfg.depth,
                actfn=self.cfg.coupling.actfn,
                mixlogcdf=False,
            )
        else:
            # TODO
            raise NotImplementedError

        self.model = VoronoiMixtureModel(
            nblocks=nblocks,
            block_fn=block_fn,
            dim=2,
            n_mixtures=self.cfg.n_mixtures,
            lazy_init=self.cfg.lazy_init,
            skip_transform=self.cfg.skip_transform,
            cond_embed_dim=self.cfg.cond_embed_dim,
        ).to(self.device)

        log.info(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss_meter = utils.RunningAverageMeter(0.99)

    def main(self):

        if self.iter == 0:
            self.initialize()
        else:
            log.info("Resuming with existing model.")

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter < self.cfg.iterations:

            with torch.autograd.set_detect_anomaly(False):

                x = inf_train_gen(self.cfg.dataset, self.cfg.batchsize)
                x = torch.as_tensor(x).float().to(self.device)

                z, delta_logp = self.model(x, logp=0)
                logpz = (
                    normal_logpdf(z, mean=0, log_std=math.log(self.cfg.prior_std))
                    .view(z.size(0), -1)
                    .sum(1, keepdim=True)
                )
                logpx = logpz - delta_logp

                loss = logpx.nanmean().mul_(-1)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            self.loss_meter.update(loss.item())

            if self.iter % self.cfg.logfreq == 0:
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter}"
                    f" | Time {time_per_itr:.4f}"
                    f" | NLL {self.loss_meter.val:.4f}({self.loss_meter.avg:.4f})"
                    f" | GradNorm {grad_norm:.4f}"
                )
                self.save()
                start_time = time.time()

            if self.iter % self.cfg.vizfreq == 0:
                self.visualize()

            self.iter += 1

        self.save()
        self.visualize(nsamples=200000, npts=400)
        self.visualize_hq(nsamples=200000, npts=400)
        log.info("Training done")

    @torch.no_grad()
    def visualize_hq(self, nsamples=200000, npts=400):
        os.makedirs("figs", exist_ok=True)

        # Data samples.
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        x = inf_train_gen(self.cfg.dataset, nsamples)
        plt_samples(x, ax, npts=npts)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.cfg.dataset}_data.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Model samples.
        # Resample z and mask.
        for m in self.model.modules():
            if isinstance(m, VoronoiTessellation):
                m.mask = None
        z = torch.randn(nsamples, 2).mul(self.cfg.prior_std).to(self.device)
        s = self.model.inverse(z).cpu().numpy()
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        x = inf_train_gen(self.cfg.dataset, nsamples)
        plt_samples(s, ax, npts=npts)

        if len(self.model.vormixes) == 1:
            line_color = "white"
            line_width = 1.5
            line_alpha = 1.0

            full_segments = self.model.vormixes[0].tessellation_edges()
            M, L, D = full_segments.shape
            full_segments = self.model.flows[0].inverse(full_segments.reshape(M * L, D))
            full_segments = full_segments.reshape(M, L, D).cpu().numpy()

            for segment in full_segments:
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    linestyle="--",
                )

            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.cfg.dataset}_samples.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Model density.
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        plt_flow_density(
            partial(normal_logpdf, mean=0, log_std=math.log(self.cfg.prior_std)),
            self.model,
            ax,
            npts=npts,
            device=self.device,
        )

        if len(self.model.vormixes) == 1:
            line_color = "white"
            line_width = 1.5
            line_alpha = 1.0

            full_segments = self.model.vormixes[0].tessellation_edges()
            M, L, D = full_segments.shape
            full_segments = self.model.flows[0].inverse(full_segments.reshape(M * L, D))
            full_segments = full_segments.reshape(M, L, D).cpu().numpy()

            for segment in full_segments:
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    linestyle="--",
                )

            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.cfg.dataset}_density.png", bbox_inches="tight", dpi=300)
        plt.close()

    @torch.no_grad()
    def visualize(self, nsamples=10000, npts=100):
        os.makedirs("figs", exist_ok=True)

        # Data samples.
        x = inf_train_gen(self.cfg.dataset, nsamples)

        # Reconstruction.
        z = self.model(torch.as_tensor(x).float().to(self.device))
        r = self.model.inverse(z).cpu().numpy()

        # Resample only z.
        z = torch.randn(nsamples, 2).mul(self.cfg.prior_std).to(self.device)
        s = self.model.inverse(z).cpu().numpy()

        # Resample z and mask.
        for m in self.model.modules():
            if isinstance(m, VoronoiTessellation):
                m.mask = None
        z = torch.randn(nsamples, 2).mul(self.cfg.prior_std).to(self.device)
        s2 = self.model.inverse(z).cpu().numpy()

        _, axs = plt.subplots(1, 5, figsize=(20, 4))
        plt_samples(x, axs[0], npts=npts)
        plt_samples(r, axs[1], npts=npts)
        plt_samples(s, axs[2], npts=npts)
        plt_samples(s2, axs[3], npts=npts)
        plt_flow_density(
            partial(normal_logpdf, mean=0, log_std=math.log(self.cfg.prior_std)),
            self.model,
            axs[4],
            npts=npts,
            device=self.device,
        )
        plt.tight_layout()
        plt.savefig(f"figs/iter-{self.iter:05d}")
        plt.close()

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


class VoronoiMixtureModel(nn.Module):
    def __init__(
        self,
        *,
        nblocks,
        block_fn,
        dim,
        n_mixtures,
        lazy_init,
        skip_transform,
        cond_embed_dim,
    ):
        super().__init__()

        self.cond_embed_dim = cond_embed_dim

        flows = []

        def _condaffine(dim):
            return layers.ConditionalAffine1d(
                dim,
                nn.Sequential(
                    nn.Linear(cond_embed_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 32),
                    nn.GELU(),
                    nn.Linear(32, dim * 2),
                ),
            )

        for k, nb_k in enumerate(nblocks):
            chain = [layers.ActNorm1d(dim)]
            if k > 0 and self.cond_embed_dim > 0:
                chain.append(_condaffine(dim))
            for i in range(nb_k):
                chain.append(
                    block_fn(
                        i,
                        (dim,),
                        cond_embed_dim=cond_embed_dim if k > 0 else 0,
                        cond_dim=1 if k > 0 else None,
                    )
                )
                chain.append(layers.ActNorm1d(dim))
                if k > 0 and self.cond_embed_dim > 0:
                    chain.append(_condaffine(dim))
            flows.append(layers.SequentialFlow(chain))

        num_mixtures = len(nblocks) - 1

        vormixes = []
        cls_embeds = []
        for i in range(num_mixtures):
            vormixes.append(
                VoronoiTessellation(
                    num_anchor_pts=n_mixtures,
                    dim=dim,
                    lazy_init=lazy_init,
                    skip_transform=skip_transform,
                )
            )
            cls_embeds.append(nn.Embedding(n_mixtures, cond_embed_dim))

        self.flows = nn.ModuleList(flows)
        self.vormixes = nn.ModuleList(vormixes)
        self.cls_embeds = nn.ModuleList(cls_embeds)

    def forward(self, x, *, logp=None):
        return_logp = logp is not None
        if logp is None:
            logp = torch.zeros(x.shape[0], 1).to(x)
        x, logp = self.flows[0](x, logp=logp)

        for vormix, cls_embed, flow in zip(
            self.vormixes, self.cls_embeds, self.flows[1:]
        ):
            x, logp, mask = vormix(x, logp=logp)

            if self.cond_embed_dim > 0:
                idx = torch.argmax(mask.to(x.dtype), dim=-1).reshape(-1)
                cond_embedding = cls_embed(idx) / np.sqrt(self.cond_embed_dim)
            else:
                cond_embedding = None

            x, logp = flow(x, logp=logp, cond=cond_embedding)

        if return_logp:
            return x, logp
        else:
            return x

    def inverse(self, z):
        for vormix, cls_embed, flow in zip(
            self.vormixes, self.cls_embeds, self.flows[1:]
        ):
            mask = vormix.sample_mask(z)

            if self.cond_embed_dim > 0:
                idx = torch.argmax(mask.to(z.dtype), dim=-1).reshape(-1)
                cond_embedding = cls_embed(idx) / np.sqrt(self.cond_embed_dim)
            else:
                cond_embedding = None

            z = flow.inverse(z, cond=cond_embedding)
            z = vormix.inverse(z, mask)

        z = self.flows[0].inverse(z)
        return z


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logpdf(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def plt_flow_density(
    prior_logdensity, inverse_transform, ax, npts=100, memory=100, device="cpu"
):
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory ** 2)):
        z_, delta_logp_ = inverse_transform(x[ii], logp=zeros[ii])
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px, extent=[-4, 4, -4, 4], origin="lower")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def plt_samples(samples, ax, npts=100):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[-4, 4], [-4, 4]], bins=npts)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


# Import like this for pickling
from train_disjoint2d import Workspace as W


@hydra.main(config_path="configs", config_name="disjoint2d")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    try:
        workspace.run()
    except Exception as e:
        workspace.save()
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
