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

setproctitle("discrete_uci")

from omegaconf import OmegaConf
import hydra

import wandb
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
import datasets
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

        try:
            assert self.cfg.use_wandb
            if "/checkpoint/" in self.work_dir:
                job_type = "sweep"
                wandb_name = "sweep_" + os.path.relpath(
                    self.work_dir,
                    f"/checkpoint/{self.cfg.user}/voronoi-dequantization/disjoint_uci/{self.cfg.dataset}",
                )
            else:
                job_type = "local"
                wandb_name = "local_" + os.path.relpath(
                    self.work_dir,
                    os.path.join(
                        self.file_dir, f"exp_local/disjoint_uci/{self.cfg.dataset}",
                    ),
                )
            os.makedirs(self.cfg.wandb.save_dir, exist_ok=True)
            wandb.init(
                entity=f"{self.cfg.user}",
                project=self.cfg.wandb.project,
                group=self.cfg.dataset,
                name=wandb_name,
                dir=self.cfg.wandb.save_dir,
                resume="allow",
                config=self.cfg,
                job_type=job_type,
            )
            log.info(f"Wandb initialized for {wandb_name}")
            self.wandb_available = True
        except Exception as e:
            log.warn(e, exc_info=True)
            log.warn("No logging to wandb available.")
            self.wandb_available = False

        dim, train_loader, val_loader, test_loader = self.load_data()

        if self.iter == 0:

            if self.cfg.block_type == "coupling":
                block_fn = partial(
                    layers.coupling_block_fn,
                    fc=True,
                    idim=self.cfg.idim,
                    zero_init=self.cfg.zero_init,
                    depth=self.cfg.depth,
                    actfn=self.cfg.actfn,
                    mixlogcdf=False,
                )
            else:
                # TODO
                raise NotImplementedError

            self.model = VoronoiMixtureModel(
                nblocks=self.cfg.nblocks,
                block_fn=block_fn,
                dim=dim,
                n_mixtures=self.cfg.n_mixtures,
                lazy_init=self.cfg.lazy_init,
                skip_transform=self.cfg.skip_transform,
                cond_embed_dim=self.cfg.cond_embed_dim,
            ).to(self.device)
            self.ema = utils.ExponentialMovingAverage(self.model)

            log.info(self.model)
            log.info(self.ema)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
            self.loss_meter = utils.RunningAverageMeter(0.99)

            self.earlystop_val_nll = 1e10
            self.earlystop_test_nll = None
        else:
            log.info("Resuming with existing model.")

        return train_loader, val_loader, test_loader

    def main(self):
        train_loader, val_loader, test_loader = self.initialize()
        train_loader = utils.inf_generator(train_loader)

        start_time = time.time()
        prev_itr = self.iter - 1

        for (x,) in train_loader:

            if self.iter >= self.cfg.iterations:
                break

            with torch.autograd.set_detect_anomaly(False):

                x = x.to(self.device)

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

                if self.iter >= self.cfg.evalfreq:
                    self.ema.apply()

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
                if self.wandb_available:
                    wandb.log(
                        {"Train NLL": self.loss_meter.val, "GradNorm": grad_norm,},
                        step=self.iter,
                    )
                self.save()
                start_time = time.time()

            if self.iter % self.cfg.evalfreq == 0 or self.iter == self.cfg.iterations:
                val_NLL = self.evaluate(val_loader, split="Val")
                test_NLL = self.evaluate(test_loader, split="Test")
                if val_NLL < self.earlystop_val_nll:
                    self.earlystop_val_nll = val_NLL
                    self.earlystop_test_nll = test_NLL
                    log.info("Updating lowest validation NLL.")
                    self.save("best")
                if self.wandb_available:
                    wandb.log(
                        {
                            "earlystop_val_nll": self.earlystop_val_nll,
                            "earlystop_test_nll": self.earlystop_test_nll,
                        },
                        step=self.iter,
                    )

            self.iter += 1

        self.save()
        log.info("Training done")

    @torch.no_grad()
    def evaluate(self, data_loader, split: str):
        self.model.eval()

        if self.iter >= self.cfg.evalfreq:
            self.ema.swap()

        eval_nll_meter = utils.AverageMeter()

        for (x,) in data_loader:
            z, delta_logp = self.model(x.to(self.device), logp=0)
            logpz = (
                normal_logpdf(z, mean=0, log_std=math.log(self.cfg.prior_std))
                .view(z.size(0), -1)
                .sum(1, keepdim=True)
            )
            logpx = logpz - delta_logp
            eval_loss = logpx.mean().mul_(-1)

            eval_nll_meter.update(eval_loss.item(), x.shape[0])

        log.info(f"Iter {self.iter}" f" | {split} NLL {eval_nll_meter.avg:.4f})")
        if self.wandb_available:
            wandb.log({f"{split} NLL": eval_nll_meter.avg}, step=self.iter)

        if self.iter >= self.cfg.evalfreq:
            self.ema.swap()

        self.model.train()
        return eval_nll_meter.avg

    def load_data(self):

        datasets.root = os.path.join(self.file_dir, "data/")

        if self.cfg.dataset == "bsds300":
            data = datasets.BSDS300()
        elif self.cfg.dataset == "power":
            data = datasets.POWER()
        elif self.cfg.dataset == "gas":
            data = datasets.GAS()
        elif self.cfg.dataset == "hepmass":
            data = datasets.HEPMASS()
        elif self.cfg.dataset == "miniboone":
            data = datasets.MINIBOONE()
        else:
            raise ValueError("Unknown dataset")

        train_set = torch.utils.data.TensorDataset(torch.from_numpy(data.trn.x).float())
        val_set = torch.utils.data.TensorDataset(torch.from_numpy(data.val.x).float())
        test_set = torch.utils.data.TensorDataset(torch.from_numpy(data.tst.x).float())

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.cfg.batchsize
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.cfg.eval_batchsize
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.cfg.eval_batchsize
        )

        return data.n_dims, train_loader, val_loader, test_loader

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


# Import like this for pickling
from train_disjoint_uci import Workspace as W


@hydra.main(config_path="configs", config_name="disjoint_uci")
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
