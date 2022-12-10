"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

from omegaconf import OmegaConf
import hydra

import numpy as np
import pickle as pkl
import random
from scipy.spatial import Voronoi
from diagnostics.voronoi_plot_2d import voronoi_plot_2d
import torch

import utils
from voronoi import VoronoiTransform


log = logging.getLogger(__name__)


class Workspace:
    def __init__(self, cfg):

        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")

    def run(self):

        log.info("Running with configuration:\n" + OmegaConf.to_yaml(self.cfg))
        self.device = torch.device("cpu")
        self.use_gpu = self.device.type == "cuda"

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        self.main()

    def main(self):
        torch.manual_seed(18611)
        self.K = 7
        vorproj = VoronoiTransform(1, self.K, 2)
        vorproj.log_scale.data.fill_(-0.8)

        with utils.random_seed_torch(1):
            self.visualize_single_projection(vorproj)
        with utils.random_seed_torch(1):
            self.visualize_cell_samples(vorproj)
        with utils.random_seed_torch(1):
            self.visualize_voronoi_mixture(vorproj)

    def visualize_single_projection(self, vorproj):
        bsz = B = 2
        N = 1
        K = self.K
        D = 2
        idx = 5

        plt.figure()
        vor = Voronoi(vorproj.anchor_pts.reshape(K, 2).detach().numpy())
        voronoi_plot_2d(
            vor,
            show_points=False,
            show_vertices=False,
            line_colors="black",
            line_width=2,
            line_alpha=1.0,
            point_size=10,
        )

        regions, vertices = voronoi_finite_polygons_2d(vor)

        # colorize
        region = regions[idx]
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.2)

        inputs = torch.randn(bsz, 1, 2) + torch.tensor([0.3, 0.0]).reshape(1, 1, 2)
        mask = torch.zeros(bsz, 1, K).bool()
        mask[:, :, idx] = True

        x_k = torch.masked_select(
            vorproj.anchor_pts.reshape(1, N, K, D), mask.reshape(B, N, K, 1)
        ).reshape(B, N, 1, D)

        z, x_k, x_lamb, lamb = vorproj.map_onto_cell(inputs, mask, return_all=True)

        x_recon, _ = vorproj.map_outside_cell(z)

        print("reconstruction error", torch.norm(x_recon - inputs))

        x_k = x_k.reshape(bsz, 2)[1]
        x = inputs.reshape(bsz, 2)[1]
        x_lamb = x_lamb.reshape(bsz, 2)[1]
        z = z.reshape(bsz, 2)[1]

        x_k = x_k.detach().numpy()
        x_lamb = x_lamb.detach().numpy()
        x = x.detach().numpy()
        z = z.detach().numpy()

        plt.plot(
            [x_k[0], x_k[0] + 100 * (x[0] - x_k[0])],
            [x_k[1], x_k[1] + 100 * (x[1] - x_k[1])],
            "k--",
            linewidth=3.0,
            alpha=0.5,
            label="Ray                 $x(\lambda)$",
        )

        plt.plot(
            x[0],
            x[1],
            marker="o",
            linestyle="None",
            markersize=15,
            label="Input               $x$",
        )
        plt.plot(
            x_k[0],
            x_k[1],
            marker="X",
            linestyle="None",
            markersize=15,
            label="Anchor pt.       $x_k$",
        )
        plt.plot(
            z[0],
            z[1],
            marker="s",
            linestyle="None",
            markersize=15,
            label="Output            $f_k(x)$",
        )
        plt.plot(
            x_lamb[0],
            x_lamb[1],
            marker="*",
            linestyle="None",
            markersize=15,
            label="Intersection    $x(\lambda^*)$",
        )

        box_constraints_max = vorproj.box_constraints_max.detach().cpu().numpy()
        box_constraints_min = vorproj.box_constraints_min.detach().cpu().numpy()
        x0, x1 = box_constraints_min[0, 0], box_constraints_max[0, 0]
        y0, y1 = box_constraints_min[0, 1], box_constraints_max[0, 1]
        plt.xlim([-1.1, 0.9])
        plt.ylim([-1.4, 1.0])

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 2, 0, 4, 3]
        plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            fontsize=18,
            loc="lower right",
            fancybox=True,
            framealpha=0.9,
            shadow=False,
            borderpad=1,
        )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig("voronoi.png")
        plt.savefig("voronoi.pdf")

    def visualize_cell_samples(self, vorproj):
        K = vorproj.num_classes
        idx = 2

        x_k = vorproj.anchor_pts[:, idx]

        # plot transformed samples
        bsz = 1000
        inputs = torch.randn(bsz, 1, 2) * 1.0 + x_k.reshape(1, 1, 2)

        mask = torch.zeros(bsz, 1, K).bool()
        mask[:, :, idx] = True

        z, x_k, x_lamb, _ = vorproj.map_onto_cell(inputs, mask, return_all=True)

        inputs_recon, _ = vorproj.map_outside_cell(z)
        print("reconstruction error", torch.norm(inputs_recon - inputs) ** 2 / 2000)

        x = inputs.reshape(bsz, 2).detach().numpy()
        x_lamb = x_lamb.reshape(bsz, 2).detach().numpy()
        z = z.reshape(bsz, 2).detach().numpy()

        plt.figure()
        vor = Voronoi(vorproj.anchor_pts.reshape(K, 2).detach().numpy())
        voronoi_plot_2d(
            vor,
            show_points=True,
            show_vertices=False,
            line_colors="black",
            line_width=2,
            line_alpha=1.0,
            point_size=10,
        )
        plt.scatter(x[:, 0], x[:, 1], s=1, color="C4")
        plt.scatter(z[:, 0], z[:, 1], s=1, color="C3")
        plt.scatter(x_lamb[:, 0], x_lamb[:, 1], s=1, color="C2")
        plt.savefig("transformed.png")

    def visualize_voronoi_mixture(self, vorproj):
        K = vorproj.num_classes
        npts = 400
        box_constraints_max = vorproj.box_constraints_max.detach().cpu().numpy()
        box_constraints_min = vorproj.box_constraints_min.detach().cpu().numpy()
        x0, x1 = box_constraints_min[0, 0], box_constraints_max[0, 0]
        y0, y1 = box_constraints_min[0, 1], box_constraints_max[0, 1]
        x_side = np.linspace(x0 + 1e-4, x1 - 1e-4, npts)
        y_side = np.linspace(y0 + 1e-4, y1 - 1e-4, npts)
        xx, yy = np.meshgrid(x_side, y_side)
        z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        z = torch.from_numpy(z).type(torch.float32)

        with torch.no_grad():
            logpz = self.compute_mixture_probability(
                z.reshape(-1, 1, 2), vorproj, std=1
            )
            logpz = logpz.cpu().detach().numpy().reshape(npts, npts)
            pz = np.exp(logpz)

        plt.figure()
        plt.hist(logpz.reshape(-1), bins=100)
        plt.savefig("prob_hist.png")

        plt.figure()
        vor = Voronoi(vorproj.anchor_pts.reshape(K, 2).detach().numpy())
        voronoi_plot_2d(
            vor,
            show_points=True,
            show_vertices=False,
            line_colors="black",
            line_width=2,
            line_alpha=1.0,
            point_size=10,
        )
        plt.contour(
            xx,
            yy,
            pz,
            levels=np.linspace(0.0001, pz.max() + 1e-5, 10),
            cmap=cm.get_cmap("GnBu", 5),
            vmin=-0.002,
        )
        plt.xlim([x0, x1])
        plt.ylim([y0, y1])
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.tight_layout()
        plt.savefig("mixture.png")
        plt.savefig("mixture.pdf")

    def compute_mixture_probability(self, z, vorproj, std=0.2):
        """
        Computes the probability within each Voronoi cell,
        after transforming from a base Gaussian centered at the anchor point.

        Inputs:
            z: (batch_size, num_discrete_variables, embedding_dim)
            vorproj: a VoronoiProjection object
            std: standard deviation of the base Gaussian
        """
        B, N, D = z.shape
        K = vorproj.num_classes
        mask = vorproj.find_nearest(z)
        x_k = torch.masked_select(
            vorproj.anchor_pts.reshape(1, N, K, D), mask.reshape(B, N, K, 1)
        ).reshape(B, N, D)

        x, logdet = vorproj.map_outside_cell(z)
        logpx = normal_logprob(x, x_k, np.log(std)).sum(-1)
        return logpx - logdet


def normal_logprob(z, mean, log_std):
    mean = torch.as_tensor(mean)
    log_std = torch.as_tensor(log_std)
    c = torch.tensor([np.log(2 * np.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# Import like this for pickling
from test_voronoi import Workspace as W


@hydra.main(config_path="configs", config_name="default")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        log.info(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
