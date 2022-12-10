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

setproctitle("train_uci")

from omegaconf import OmegaConf
import hydra

import git
import logging
import os
import time
import math
import random
import numpy as np

import pickle as pkl

import json
import wandb

import torch
import torch.optim as optim

import discrete_datasets
from model_utils import ConditionalSimplexFlowDistribution, SimplexFlowDistribution
from model_utils import ConditionalVoronoiFlowDistribution, VoronoiFlowDistribution
from model_utils import ConditionalArgmaxFlowDistribution
from model_utils import (
    BaseGaussianDistribution,
    ConditionalGaussianDistribution,
    ResampledGaussianDistribution,
)
from model_utils import MLP
import utils
from utils import StupidNaNsException
from voronoi import VoronoiTransform

log = logging.getLogger(__name__)


def compute_logpdf(
    model, vardeq_model, discrete_samples, num_samples=1, device="cpu", bpd=False
):
    n = discrete_samples.shape[0]
    discrete_samples = discrete_samples.repeat_interleave(num_samples, dim=0).to(device)
    samples, dequantization_logpdf = vardeq_model.sample(
        discrete_samples, return_logpdf=True
    )

    logprob = model(samples)

    if ~torch.isfinite(logprob).all():
        log.warn(
            "Found {} samples with NaN or Inf log probability from model.".format(
                (~torch.isfinite(logprob)).sum().item()
            )
        )

    avg_model_logprob = logprob.mean()
    avg_dequant_logprob = dequantization_logpdf.mean()

    logprob = logprob - dequantization_logpdf

    if ~torch.isfinite(dequantization_logpdf).all():
        log.warn(
            "Found {} samples with NaN or Inf log probability from dequant model.".format(
                (~torch.isfinite(dequantization_logpdf)).sum().item()
            )
        )

    logprob = logprob.reshape(n, num_samples, 1)
    logprob = torch.logsumexp(logprob, dim=1) - math.log(num_samples)

    if bpd:
        # Convert to bits per dimension.
        logprob = logprob / (math.log(2) * discrete_samples.shape[1])
        avg_model_logprob = avg_model_logprob / (
            math.log(2) * discrete_samples.shape[1]
        )
        avg_dequant_logprob = avg_dequant_logprob / (
            math.log(2) * discrete_samples.shape[1]
        )

    return logprob, avg_model_logprob, avg_dequant_logprob

def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = base_learning_rate
    return learning_rate


def set_learning_rate(optimizer, lr):
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = lr


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)

        self.iter = 0

    def run(self):

        repo = git.Repo(self.file_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        log.info(repo)
        log.info(f"Latest commit is {sha}")
        log.info(f"Files modified from latest commit are:")
        for item in repo.index.diff(None):
            log.info(f"{item.a_path}")
        log.info("----")

        try:
            assert False  # disable wandb
            assert "exp/" in self.work_dir
            if os.path.isfile(os.path.join(self.file_dir, ".wandb.key.json")):
                with open(os.path.join(self.file_dir, ".wandb.key.json")) as f:
                    data = json.load(f)
                wandb_apikey = data.get("wandbapikey")
                os.environ["WANDB_API_KEY"] = wandb_apikey
                wandb_name = os.path.relpath(
                    self.work_dir,
                    os.path.join(
                        self.file_dir, f"exp/uci_categorical/{self.cfg.dataset}"
                    ),
                )
                wandb.init(
                    project=f"vq-uci-{self.cfg.dataset}",
                    entity="rtqichen",
                    name=wandb_name,
                    resume="allow",
                )
                log.info(f"Wandb initialized for {wandb_name}")
                self.wandb_available = True
        except Exception as e:
            log.warn("No logging to wandb available.")
            self.wandb_available = False
            pass

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
            torch.backends.cudnn.benchmark = True
        else:
            log.info("WARNING: Using device {}".format(device))

        self.device = device
        self.use_gpu = device.type == "cuda"

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        try:
            dataloader = self.initialize()
            self.main(dataloader)
        finally:
            if self.wandb_available:
                wandb.finish()

    def initialize(self):
        if self.cfg.dataset == "mushroom":
            dataset_cls = discrete_datasets.Mushroom
        elif self.cfg.dataset == "nursery":
            dataset_cls = discrete_datasets.Nursery
        elif self.cfg.dataset == "connect4":
            dataset_cls = discrete_datasets.Connect4
        elif self.cfg.dataset == "uscensus90":
            dataset_cls = discrete_datasets.USCensus90
        elif self.cfg.dataset == "pokerhand":
            dataset_cls = discrete_datasets.PokerHand
        elif self.cfg.dataset == "forests":
            dataset_cls = discrete_datasets.Forests
        elif self.cfg.dataset == "text8":
            dataset_cls = discrete_datasets.Text8
        elif self.cfg.dataset == "enwik8":
            dataset_cls = discrete_datasets.EnWik8
        else:
            raise ValueError(f"Unknown dataset {self.cfg.dataset}")

        train_ds = dataset_cls(root=self.file_dir, split="train")
        val_ds = dataset_cls(
            root=self.file_dir, split="val", raw_dicts=train_ds.raw_dicts,
        )
        test_ds = dataset_cls(
            root=self.file_dir, split="test", raw_dicts=train_ds.raw_dicts,
        )

        log.info(f"There are {len(train_ds.K)} variables with {max(train_ds.K)} values")
        self.num_classes = max(train_ds.K)
        self.num_discrete_variables = len(train_ds.K)

        log.info(f"Train set has {len(train_ds)} examples")
        log.info(f"Val set has {len(val_ds)} examples")
        log.info(f"Test set has {len(test_ds)} examples")

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.cfg.eval_batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.cfg.eval_batch_size, shuffle=False
        )

        data_loaders = (train_loader, val_loader, test_loader)

        if self.iter > 0:
            log.info("Resuming from existing model.")
            return data_loaders

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
                self.num_discrete_variables,
                self.num_classes - 1,
                self.cfg.num_flows,
                self.cfg.hdims,
                base=model_base_dist(
                    d=self.num_discrete_variables * (self.num_classes - 1)
                ),
                actfn=self.cfg.actfn,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                flow_type=self.cfg.flow_type,
            ).to(self.device)

            self.vardeq_model = ConditionalSimplexFlowDistribution(
                self.num_discrete_variables,
                self.num_classes - 1,
                self.cfg.use_dequant_flow,
                self.cfg.num_dequant_flows,
                self.cfg.hdims,
                base=BaseGaussianDistribution(
                    d=self.num_discrete_variables * (self.num_classes - 1)
                ),
                actfn=self.cfg.actfn,
                cond_embed_dim=self.cfg.cond_embed_dim,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                use_contextnet=self.cfg.use_contextnet,
                flow_type=self.cfg.flow_type,
            ).to(self.device)
            self.all_parameters = set(
                list(self.model.parameters()) + list(self.vardeq_model.parameters())
            )

        elif self.cfg.dequantization == "voronoi":

            voronoi_transform = VoronoiTransform(
                self.num_discrete_variables,
                self.num_classes,
                self.cfg.embedding_dim,
                share_embeddings=self.cfg.share_embeddings,
                learn_box_constraints=self.cfg.learn_box_constraints,
            )

            self.model = VoronoiFlowDistribution(
                voronoi_transform,
                self.num_discrete_variables,
                self.cfg.embedding_dim,
                self.cfg.num_flows,
                self.cfg.hdims,
                base=model_base_dist(
                    d=self.num_discrete_variables * self.cfg.embedding_dim
                ),
                actfn=self.cfg.actfn,
                use_logit_transform=self.cfg.use_logit_transform,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                flow_type=self.cfg.flow_type,
            ).to(self.device)

            self.vardeq_model = ConditionalVoronoiFlowDistribution(
                voronoi_transform,
                self.num_discrete_variables,
                self.num_classes,
                self.cfg.embedding_dim,
                self.cfg.use_dequant_flow,
                self.cfg.num_dequant_flows,
                self.cfg.hdims,
                base=ConditionalGaussianDistribution(
                    d=self.num_discrete_variables * self.cfg.embedding_dim,
                    cond_embed_dim=self.num_discrete_variables
                    * self.cfg.cond_embed_dim,
                    actfn=self.cfg.actfn,
                ),
                actfn=self.cfg.actfn,
                cond_embed_dim=self.cfg.cond_embed_dim,
                share_embeddings=self.cfg.share_embeddings,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                use_contextnet=self.cfg.use_contextnet,
                flow_type=self.cfg.flow_type,
            ).to(self.device)
            self.all_parameters = set(
                list(self.model.parameters()) + list(self.vardeq_model.parameters())
            )

        elif self.cfg.dequantization == "argmax":
            embedding_dim = int(np.ceil(np.log2(self.num_classes)))

            self.model = VoronoiFlowDistribution(
                None,
                self.num_discrete_variables,
                embedding_dim,
                self.cfg.num_flows,
                self.cfg.hdims,
                base=model_base_dist(d=self.num_discrete_variables * embedding_dim),
                actfn=self.cfg.actfn,
                use_logit_transform=False,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                flow_type=self.cfg.flow_type,
            ).to(self.device)

            self.vardeq_model = ConditionalArgmaxFlowDistribution(
                self.num_discrete_variables,
                self.num_classes,
                self.cfg.use_dequant_flow,
                self.cfg.num_dequant_flows,
                self.cfg.hdims,
                base=ConditionalGaussianDistribution(
                    d=self.num_discrete_variables * embedding_dim,
                    cond_embed_dim=self.num_discrete_variables
                    * self.cfg.cond_embed_dim,
                    actfn=self.cfg.actfn,
                ),
                actfn=self.cfg.actfn,
                cond_embed_dim=self.cfg.cond_embed_dim,
                share_embeddings=self.cfg.share_embeddings,
                arch=self.cfg.arch,
                num_transformer_layers=self.cfg.num_transformer_layers,
                transformer_d_model=self.cfg.transformer_d_model,
                transformer_dropout=self.cfg.transformer_dropout,
                block_transform=self.cfg.block_transform,
                num_mixtures=self.cfg.num_mixtures,
                use_contextnet=self.cfg.use_contextnet,
                flow_type=self.cfg.flow_type,
            ).to(self.device)

            self.all_parameters = set(
                list(self.model.parameters()) + list(self.vardeq_model.parameters())
            )
        else:
            raise ValueError(f"Unknown dequantization method {self.cfg.dequantization}")

        self.ema_model = utils.ExponentialMovingAverage(self.model)
        self.ema_vardeq_model = utils.ExponentialMovingAverage(self.vardeq_model)

        self.optimizer = optim.AdamW(
            self.all_parameters, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        log.info("--- Model ---")
        log.info(self.model)
        log.info(self.ema_model)
        log.info("--- Dequant Model ---")
        log.info(self.vardeq_model)
        log.info(self.ema_vardeq_model)

        self.loss_meter = utils.RunningAverageMeter(0.99)
        self.model_logprob_meter = utils.RunningAverageMeter(0.99)
        self.dequant_logprob_meter = utils.RunningAverageMeter(0.99)

        self.earlystop_val_nll = 1e10
        self.earlystop_test_nll = None

        return data_loaders

    def main(self, data_loaders):
        train_loader, val_loader, test_loader = data_loaders
        train_data_generator = utils.inf_generator(train_loader)

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter < self.cfg.iterations:

            lr = learning_rate_schedule(self.iter, self.cfg.warmup, self.cfg.lr, self.cfg.iterations)
            set_learning_rate(self.optimizer, lr)

            batch = next(train_data_generator)[0]
            logprob, avg_model_logprob, avg_dequant_logprob = compute_logpdf(
                self.model,
                self.vardeq_model,
                batch,
                device=self.device,
                # bpd=self.cfg.dataset in ["text8", "enwik8"],
                bpd=False,
            )

            loss = logprob[torch.isfinite(logprob)].mean().mul_(-1)

            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.all_parameters, 1e20)

            if torch.isnan(grad_norm):
                raise StupidNaNsException("NaN gradient.")

            self.optimizer.step()

            if self.iter >= self.cfg.iterations // 2:
                self.ema_model.apply()
                self.ema_vardeq_model.apply()

            self.loss_meter.update(loss.item())
            self.model_logprob_meter.update(avg_model_logprob.item())
            self.dequant_logprob_meter.update(avg_dequant_logprob.item())

            if self.iter % self.cfg.logfreq == 0:
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter}"
                    f" | Time {time_per_itr:.4f}"
                    f" | Train NLL {self.loss_meter.val:.4f}({self.loss_meter.avg:.4f})"
                    f" | Model Logprob {self.model_logprob_meter.val:.4f}({self.model_logprob_meter.avg:.4f})"
                    f" | Dequant Logprob {self.dequant_logprob_meter.val:.4f}({self.dequant_logprob_meter.avg:.4f})"
                    f" | GradNorm {grad_norm:.4f}"
                )
                self.save()
                if self.wandb_available:
                    wandb.log(
                        {
                            "Iter": self.iter,
                            "Train NLL": self.loss_meter.val,
                            "Model Logprob": self.model_logprob_meter.val,
                            "Dequant Logprob": self.dequant_logprob_meter.val,
                            "GradNorm": grad_norm,
                        }
                    )
                start_time = time.time()

            if (
                not self.cfg.skip_eval
                and self.iter > 0
                and self.iter % self.cfg.evalfreq == 0
            ):
                val_NLL = self.evaluate(val_loader, split="Val")
                test_NLL = self.evaluate(test_loader, split="Test")
                if val_NLL < self.earlystop_val_nll:
                    self.earlystop_val_nll = val_NLL
                    self.earlystop_test_nll = test_NLL
                    log.info("Updating lowest validation NLL.")
                    self.save("best")

            self.iter += 1

    @torch.no_grad()
    def evaluate(self, dataloader, split):

        if self.cfg.ema_eval and self.iter >= self.cfg.iterations // 2:
            self.ema_model.swap()
            self.ema_vardeq_model.swap()
        self.model.eval()
        self.vardeq_model.eval()

        eval_nll_meter = utils.AverageMeter()
        eval_model_logprob_meter = utils.AverageMeter()
        eval_dequant_logprob_meter = utils.AverageMeter()

        for batch in dataloader:
            batch = batch[0]

            eval_logprob, eval_model_logprob, eval_dequant_logprob = compute_logpdf(
                self.model,
                self.vardeq_model,
                batch,
                num_samples=self.cfg.num_eval_samples,
                device=self.device,
                # bpd=self.cfg.dataset in ["text8", "enwik8"],
                bpd=False,
            )
            eval_loss = eval_logprob.mean().mul_(-1)

            eval_nll_meter.update(eval_loss.item(), batch.shape[0])
            eval_model_logprob_meter.update(eval_model_logprob.item(), batch.shape[0])
            eval_dequant_logprob_meter.update(
                eval_dequant_logprob.item(), batch.shape[0]
            )

        log.info(
            f"Iter {self.iter}"
            f" | {split} NLL {eval_nll_meter.avg:.4f})"
            f" | Model Logprob {eval_model_logprob_meter.avg:.4f})"
            f" | Dequant Logprob {eval_dequant_logprob_meter.avg:.4f})"
        )
        if self.wandb_available:
            wandb.log(
                {
                    "Iter": self.iter,
                    f"{split} NLL": eval_nll_meter.avg,
                    f"{split} Model Logprob": eval_model_logprob_meter.avg,
                    f"{split} Dequant Logprob": eval_dequant_logprob_meter.avg,
                }
            )

        if self.cfg.ema_eval and self.iter >= self.cfg.iterations // 2:
            self.ema_model.swap()
            self.ema_vardeq_model.swap()
        self.model.train()
        self.vardeq_model.train()
        return eval_nll_meter.avg

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


# Import like this for pickling
from train_uci import Workspace as W


def run_with_automatic_resume(workspace, fname, n_resumes):
    try:
        workspace.run()
    except StupidNaNsException as e:
        if n_resumes <= 0:
            log.error("Too many resumes due to NaN gradient.")
            raise RuntimeError("Too many resumes due to NaN gradient.")
        del workspace

        log.info(f"Caught StupidNaNsException. Resuming from {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)

        run_with_automatic_resume(workspace, fname, n_resumes - 1)


@hydra.main(config_path="configs", config_name="uci_categorical")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        log.info(f"Resuming from {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
        workspace.work_dir = os.getcwd()
    else:
        workspace = W(cfg)

    try:
        run_with_automatic_resume(workspace, fname, n_resumes=cfg.n_resumes)
    except Exception as e:
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
