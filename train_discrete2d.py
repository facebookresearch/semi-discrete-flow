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

import logging
import os
import time
import math
import random
from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pickle as pkl

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from model_utils import ConditionalSimplexFlowDistribution, SimplexFlowDistribution
from model_utils import ConditionalVoronoiFlowDistribution, VoronoiFlowDistribution
from model_utils import (
    BaseGaussianDistribution,
    ConditionalGaussianDistribution,
    ResampledGaussianDistribution,
)
from model_utils import logsimplex_uniform_dequantize
from model_utils import MLP
import utils
from voronoi import VoronoiTransform

log = logging.getLogger(__name__)


def get_joint_pmf(dataset, file_dir=None):
    def cluster(pmf, i, j):
        pmf[i, j] = 1
        pmf[i - 1, j] = 0.25
        pmf[i + 1, j] = 0.25
        pmf[i, j - 1] = 0.25
        pmf[i, j + 1] = 0.25

    if dataset == "center3":
        joint_pmf = np.zeros((3, 3))
        joint_pmf[1, 1] = 1
        return joint_pmf

    elif dataset == "cluster3":
        joint_pmf = np.zeros((3, 3))
        cluster(joint_pmf, 1, 1)
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "diagonal4":
        joint_pmf = np.eye(4)
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "cluster10":
        joint_pmf = np.zeros((10, 10))
        cluster(joint_pmf, 2, 2)
        cluster(joint_pmf, 2, 7)
        cluster(joint_pmf, 7, 2)
        cluster(joint_pmf, 7, 7)
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "cross10":
        joint_pmf = np.eye(10)
        joint_pmf = joint_pmf + joint_pmf[::-1]
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "diagonal28":
        joint_pmf = np.eye(28)
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "diamond28":
        joint_pmf = np.zeros((20, 20))
        joint_pmf[1::3, 0::3] = 1
        joint_pmf[0::3, 1::3] = 1
        joint_pmf = ndimage.rotate(joint_pmf, 45, reshape=True, order=1)
        joint_pmf = joint_pmf / joint_pmf.sum()
        return joint_pmf

    elif dataset == "discrete_8gaussians":
        joint_pmf = np.loadtxt(
            os.path.join(file_dir, "data/discrete_8gaussians_pmf.csv"), delimiter=","
        )
        return joint_pmf

    elif dataset == "discrete_pinwheel":
        joint_pmf = np.loadtxt(
            os.path.join(file_dir, "data/discrete_pinwheel_pmf.csv"), delimiter=","
        )
        return joint_pmf

    elif dataset == "voronoy":
        joint_pmf = np.loadtxt(
            os.path.join(file_dir, "data/voronoy_pmf.csv"), delimiter=","
        )
        return joint_pmf

    else:
        raise ValueError(f"Unknown dataset option {dataset}")


def sample_from_joint_pmf(joint_pmf, shape, device):
    if isinstance(shape, int):
        shape = (shape,)
    h, w = joint_pmf.shape
    probs = joint_pmf.reshape(-1)
    sample = torch.multinomial(probs, int(np.prod(shape)), replacement=True)
    x = torch.div(sample, w, rounding_mode="floor")
    y = sample % w
    return torch.stack([x, y], axis=-1).reshape(*shape, 2)


def samples_to_freq(samples, K):
    x, y = samples[:, 0], samples[:, 1]
    indices = x * K + y
    counts = np.zeros(K * K)
    for i in range(K * K):
        counts[i] = (indices == i).sum()
    freq = counts / counts.sum()
    freq = freq.reshape(K, K)
    return freq


def compute_logpdf_simplex(
    model, vardeq_model, discrete_samples, K, num_samples=1, device="cpu"
):
    n = discrete_samples.shape[0]
    discrete_samples = discrete_samples.repeat_interleave(num_samples, dim=0).to(device)
    if vardeq_model is None:
        samples, dequantization_logpdf = logsimplex_uniform_dequantize(
            discrete_samples, K
        )
        dequantization_logpdf = dequantization_logpdf.sum(1, keepdim=True)
    else:
        samples, dequantization_logpdf = vardeq_model.sample(
            discrete_samples, logits=True, return_logpdf=True
        )

    logprob = model(samples, logits=True)

    avg_model_logprob = logprob.mean()
    avg_dequant_logprob = dequantization_logpdf.mean()

    logprob = logprob - dequantization_logpdf
    logprob = logprob.reshape(n, num_samples, 1)
    logprob = torch.logsumexp(logprob, dim=1) - math.log(num_samples)
    return logprob, avg_model_logprob, avg_dequant_logprob


def compute_logpdf(model, vardeq_model, discrete_samples, num_samples=1, device="cpu"):
    n = discrete_samples.shape[0]
    discrete_samples = discrete_samples.repeat_interleave(num_samples, dim=0).to(device)
    samples, dequantization_logpdf = vardeq_model.sample(
        discrete_samples, return_logpdf=True
    )

    if torch.isnan(dequantization_logpdf).any():
        log.info(f"Found NaN in dequantization probability.")

    logprob = model(samples)

    if torch.isnan(logprob).any():
        log.info(f"Found NaN in model probability.")

    avg_model_logprob = logprob.nanmean()
    avg_dequant_logprob = dequantization_logpdf.nanmean()

    logprob = logprob - dequantization_logpdf

    if torch.isnan(logprob).any():
        raise RuntimeError(
            "Found {} samples with NaN log probability.".format(
                torch.isnan(logprob).sum().item()
            )
        )

    logprob = logprob.reshape(n, num_samples, 1)
    logprob = torch.logsumexp(logprob, dim=1) - math.log(num_samples)
    return logprob, avg_model_logprob, avg_dequant_logprob


def flatten(discrete_samples):
    K = discrete_samples.shape[-1]
    x, y = discrete_samples[:, 0], discrete_samples[:, 1]
    joint_idx = x * K + y
    return joint_idx.reshape(-1, 1)


def entropy(joint_pmf):
    entropy = 0.0
    for p in joint_pmf.reshape(-1):
        if p > 0:
            entropy -= p * torch.log(p)
    return entropy


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

        self.joint_pmf = torch.as_tensor(
            get_joint_pmf(self.cfg.dataset, self.file_dir)
        ).to(self.device)

        self.num_classes = self.joint_pmf.shape[-1]
        self.num_discrete_variables = 2

        if self.cfg.flatten:
            self.num_classes = self.joint_pmf.shape[-1] * self.joint_pmf.shape[-1]
            self.num_discrete_variables = 1

        os.makedirs(os.path.join(self.work_dir, "imgs"), exist_ok=True)
        self.save_pmf(
            self.joint_pmf.cpu().numpy(),
            os.path.join(self.work_dir, "imgs", "target_density.png"),
        )

        def model_base_dist(d):
            if self.cfg.base == "gaussian":
                return BaseGaussianDistribution(d)
            elif self.cfg.base == "resampled":
                net_a = MLP(
                    [d] + list(self.cfg.resampled.hdims) + [1], self.cfg.resampled.actfn
                )
                return ResampledGaussianDistribution(d, net_a, 100, 0.1)
            else:
                raise ValueError(f"Unknown base option {self.cfg.base}")

        if self.cfg.dequantization == "simplex":
            self.model = SimplexFlowDistribution(
                num_discrete_variables=self.num_discrete_variables,
                embedding_dim=self.num_classes - 1,
                num_blocks=self.cfg.num_flows,
                hdims=self.cfg.hdims,
                base=model_base_dist(
                    d=self.num_discrete_variables * (self.num_classes - 1)
                ),
                actfn=self.cfg.actfn,
            ).to(self.device)

            if self.cfg.vardeq:
                self.vardeq_model = ConditionalSimplexFlowDistribution(
                    num_discrete_variables=self.num_discrete_variables,
                    embedding_dim=self.num_classes - 1,
                    use_dequant_flow=True,
                    num_blocks=self.cfg.num_dequant_flows,
                    hdims=self.cfg.hdims,
                    base=BaseGaussianDistribution(
                        d=self.num_discrete_variables * (self.num_classes - 1)
                    ),
                    actfn=self.cfg.actfn,
                ).to(self.device)
                self.all_parameters = set(
                    list(self.model.parameters()) + list(self.vardeq_model.parameters())
                )
            else:
                self.vardeq_model = None
                self.all_parameters = set(list(self.model.parameters()))
        elif self.cfg.dequantization == "voronoi":

            voronoi_transform = VoronoiTransform(
                self.num_discrete_variables,
                self.num_classes,
                self.cfg.embedding_dim,
                learn_box_constraints=False,
            )

            self.model = VoronoiFlowDistribution(
                voronoi_transform=voronoi_transform,
                num_discrete_variables=self.num_discrete_variables,
                embedding_dim=self.cfg.embedding_dim,
                num_blocks=self.cfg.num_flows,
                hdims=self.cfg.hdims,
                base=model_base_dist(
                    d=self.num_discrete_variables * self.cfg.embedding_dim
                ),
                actfn=self.cfg.actfn,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
            ).to(self.device)

            if self.cfg.vardeq:
                self.vardeq_model = ConditionalVoronoiFlowDistribution(
                    voronoi_transform=voronoi_transform,
                    num_discrete_variables=self.num_discrete_variables,
                    num_classes=self.num_classes,
                    embedding_dim=self.cfg.embedding_dim,
                    use_dequant_flow=False,
                    num_blocks=self.cfg.num_dequant_flows,
                    hdims=self.cfg.hdims,
                    base=ConditionalGaussianDistribution(
                        d=self.num_discrete_variables * self.cfg.embedding_dim,
                        cond_embed_dim=self.num_discrete_variables
                        * self.cfg.cond_embed_dim,
                        actfn=self.cfg.actfn,
                    ),
                    actfn=self.cfg.actfn,
                    cond_embed_dim=self.cfg.cond_embed_dim,
                    block_transform=self.cfg.block_transform,
                    num_mixtures=self.cfg.num_mixtures,
                ).to(self.device)
                self.all_parameters = set(
                    list(self.model.parameters()) + list(self.vardeq_model.parameters())
                )
            else:
                raise ValueError(
                    "Option `vardeq` must be True for voronoi dequantization."
                )
        else:
            raise ValueError(f"Unknown dequantization method {self.cfg.dequantization}")

        print(self.model)
        print(self.vardeq_model)

        self.optimizer = optim.Adam(self.all_parameters, lr=self.cfg.lr)

        self.loss_meter = utils.RunningAverageMeter(0.99)
        self.model_logprob_meter = utils.RunningAverageMeter(0.9)
        self.dequant_logprob_meter = utils.RunningAverageMeter(0.9)

    def main(self):

        if self.iter == 0:
            self.initialize()
        else:
            log.info("Resuming with existing model.")

        if self.joint_pmf.shape[0] < 10:
            probs = self.joint_pmf
            log.info(
                "\n"
                + "\n".join(
                    [
                        "  ".join(["{:.3f}".format(item) for item in row])
                        for row in probs
                    ]
                )
            )
        log.info(f"Target entropy: {entropy(self.joint_pmf).item()}")

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter < self.cfg.iterations:

            with torch.autograd.set_detect_anomaly(False):

                discrete_samples = sample_from_joint_pmf(
                    self.joint_pmf, self.cfg.batch_size, self.device
                )
                if self.cfg.flatten:
                    discrete_samples = flatten(discrete_samples)

                if self.cfg.dequantization == "simplex":
                    (
                        logprob,
                        avg_model_logprob,
                        avg_dequant_logprob,
                    ) = compute_logpdf_simplex(
                        self.model,
                        self.vardeq_model,
                        discrete_samples,
                        self.num_classes - 1,
                        num_samples=1,
                        device=self.device,
                    )
                else:
                    logprob, avg_model_logprob, avg_dequant_logprob = compute_logpdf(
                        self.model,
                        self.vardeq_model,
                        discrete_samples,
                        num_samples=1,
                        device=self.device,
                    )

                loss = logprob.nanmean().mul_(-1)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.all_parameters, 5.0)
                self.optimizer.step()

            self.loss_meter.update(loss.item())
            self.model_logprob_meter.update(avg_model_logprob.item())
            self.dequant_logprob_meter.update(avg_dequant_logprob.item())

            if (self.iter + 1) % 200 == 0:
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter + 1}"
                    f" | Time {time_per_itr:.4f}"
                    f" | NLL {self.loss_meter.val:.4f}({self.loss_meter.avg:.4f})"
                    f" | Model Logprob {self.model_logprob_meter.val:.4f}({self.model_logprob_meter.avg:.4f})"
                    f" | Dequant Logprob {self.dequant_logprob_meter.val:.4f}({self.dequant_logprob_meter.avg:.4f})"
                    f" | GradNorm {grad_norm:.4f}"
                )
                self.evaluate()
                self.save()
                start_time = time.time()

            if (self.iter + 1) % 2500 == 0:
                self.visualize()

            self.iter += 1

        self.save()
        self.visualize_samples_hq()
        log.info("Training done")

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()
        self.vardeq_model.eval()

        test_nll_meter = utils.AverageMeter()
        test_model_logprob_meter = utils.AverageMeter()
        test_dequant_logprob_meter = utils.AverageMeter()

        for _ in range(20):
            discrete_samples = sample_from_joint_pmf(
                self.joint_pmf, self.cfg.test_batch_size, self.device
            )

            if self.cfg.dequantization == "simplex":
                (
                    test_logprob,
                    test_model_logprob,
                    test_dequant_logprob,
                ) = compute_logpdf_simplex(
                    self.model,
                    self.vardeq_model,
                    discrete_samples,
                    self.num_classes - 1,
                    num_samples=self.cfg.num_test_samples,
                    device=self.device,
                )
            else:
                test_logprob, test_model_logprob, test_dequant_logprob = compute_logpdf(
                    self.model,
                    self.vardeq_model,
                    discrete_samples,
                    num_samples=self.cfg.num_test_samples,
                    device=self.device,
                )
            test_loss = test_logprob.mean().mul_(-1)

            test_nll_meter.update(test_loss.item(), discrete_samples.shape[0])
            test_model_logprob_meter.update(
                test_model_logprob.item(), discrete_samples.shape[0]
            )
            test_dequant_logprob_meter.update(
                test_dequant_logprob.item(), discrete_samples.shape[0]
            )

        log.info(
            f"Iter {self.iter}"
            f" | Test NLL {test_nll_meter.avg:.4f})"
            f" | Model Logprob {test_model_logprob_meter.avg:.4f})"
            f" | Dequant Logprob {test_dequant_logprob_meter.avg:.4f})"
        )
        self.model.train()
        self.vardeq_model.train()

    def plot_embedding(self, z, label):
        if self.cfg.dequantization != "voronoi" or self.cfg.embedding_dim != 2:
            return

        voronoi_transform = self.model.voronoi_transform
        num_classes = voronoi_transform.num_classes

        z = z.cpu().detach().numpy()
        anchor_pts = (
            self.model.voronoi_transform.anchor_pts.reshape(
                self.num_discrete_variables, num_classes, -1
            )[:, :, :2]
            .cpu()
            .detach()
            .numpy()
        )

        for i in range(self.num_discrete_variables):
            plt.figure()
            vor = Voronoi(anchor_pts[i])
            voronoi_plot_2d(
                vor,
                show_vertices=False,
                line_colors="orange",
                line_width=2,
                line_alpha=0.6,
                point_size=10,
            )
            plt.scatter(z[:, i, 0], z[:, i, 1], s=1, color="C3")
            os.makedirs(os.path.join(self.work_dir, "imgs"), exist_ok=True)
            plt.xlim([-1.1, 1.1])
            plt.ylim([-1.1, 1.1])
            plt.savefig(
                os.path.join(
                    self.work_dir,
                    "imgs",
                    f"{label}_samples_iter{self.iter:06d}_dim{i}.png",
                )
            )
            plt.close()
        plt.close("all")

    def plot_discrete_samples_hist(self, discrete_samples, label):
        data_prob_table = np.histogramdd(
            discrete_samples.cpu().detach().numpy(), bins=self.num_classes
        )
        self.save_pmf(
            data_prob_table[0] / np.sum(data_prob_table[0]),
            os.path.join(
                self.work_dir,
                "imgs",
                f"{label}_discrete_samples_iter{self.iter:06d}.png",
            ),
        )

    def visualize_samples_hq(self):
        self.model.eval()
        self.vardeq_model.eval()
        if self.cfg.dequantization == "voronoi":
            with torch.no_grad():

                all_model_discrete_samples = []

                for _ in tqdm(range(10)):
                    z = self.model.sample(1000000)
                    model_discrete_samples_one_hot = self.model.voronoi_transform.find_nearest(
                        z
                    )
                    model_discrete_samples = torch.argmax(
                        model_discrete_samples_one_hot.float(), dim=-1
                    )
                    all_model_discrete_samples.append(model_discrete_samples.cpu())
                model_discrete_samples = torch.cat(all_model_discrete_samples, dim=0)
                self.plot_discrete_samples_hist(model_discrete_samples, "model_final")
                # self.plot_embedding(z, "model_final")

    def visualize(self):
        self.model.eval()
        self.vardeq_model.eval()

        # Plot samples from the dequantization model and the density model.
        if self.cfg.dequantization == "voronoi":
            with torch.no_grad():
                discrete_samples = sample_from_joint_pmf(
                    self.joint_pmf, 10000, self.device
                )
                z = self.vardeq_model.sample(discrete_samples)
                self.plot_discrete_samples_hist(discrete_samples, "dequant")
                self.plot_embedding(z, "dequant")

                z = self.model.sample(10000)
                model_discrete_samples_one_hot = self.model.voronoi_transform.find_nearest(
                    z
                )
                model_discrete_samples = torch.argmax(
                    model_discrete_samples_one_hot.float(), dim=-1
                )
                self.plot_discrete_samples_hist(model_discrete_samples, "model")
                self.plot_embedding(z, "model")

        if self.num_discrete_variables == 1:
            K = int(math.sqrt(self.num_classes))
        elif self.num_discrete_variables == 2:
            K = self.num_classes

        if K < 20:

            joint_idx = torch.arange(K * K).reshape(-1, 1)

            if self.num_discrete_variables == 2:
                x = torch.div(joint_idx, K, rounding_mode="floor")
                y = joint_idx % K
                joint_idx = torch.stack([x, y], axis=-1).reshape(-1, 2)

            with torch.no_grad():
                if self.cfg.dequantization == "simplex":
                    logprob, _, _ = compute_logpdf_simplex(
                        self.model,
                        self.vardeq_model,
                        joint_idx,
                        K - 1,
                        num_samples=10,
                        device=self.device,
                    )
                else:
                    logprob, _, _ = compute_logpdf(
                        self.model,
                        self.vardeq_model,
                        joint_idx,
                        num_samples=10,
                        device=self.device,
                    )

            logprob = logprob.reshape(K, K)

            if logprob.shape[0] < 10:
                probs = torch.exp(logprob)
                log.info(
                    "\n"
                    + "\n".join(
                        [
                            "  ".join(["{:.3f}".format(item) for item in row])
                            for row in probs
                        ]
                    )
                )

            sumprobs = torch.exp(torch.logsumexp(logprob.reshape(-1), 0)).item()
            log.info(f"Sum of model probabilities: {sumprobs}")

            self.save_pmf(
                logprob.exp().detach().cpu().numpy(),
                os.path.join(self.work_dir, "imgs", f"pmf_iter{self.iter:06d}.png"),
            )

        self.model.train()
        self.vardeq_model.train()

    def save_pmf(self, data, filename):
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(
            data,
            cmap=cm.get_cmap("Blues", 6),
            extent=[0, self.num_classes, 0, self.num_classes],
            interpolation="nearest",
        )
        fig.savefig(filename, dpi=data.shape[0] * 4)
        plt.close(fig)

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


# Import like this for pickling
from train_discrete2d import Workspace as W


@hydra.main(config_path="configs", config_name="discrete2d")
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
