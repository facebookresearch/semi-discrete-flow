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

import ipdb
from tqdm import tqdm
import math
import git
import logging
import os
import time
import random
import numpy as np

import pickle as pkl

import torch
import torch.optim as optim

import discrete_datasets
import layers
import utils
from utils import StupidNaNsException
from model_itemsets import DeterminantalPointProcesses
from model_itemsets import DiscreteSetsFlow

log = logging.getLogger(__name__)


def compute_loglik(model, discrete_samples, num_samples=1):
    n = discrete_samples.shape[0]
    discrete_samples = discrete_samples.repeat_interleave(num_samples, dim=0)
    loglik = model(discrete_samples)
    loglik = loglik.reshape(n, num_samples)
    loglik = torch.logsumexp(loglik, dim=1) - math.log(num_samples)
    return loglik


def count_nfe(model):
    nfe = 0
    for m in model.modules():
        if isinstance(m, layers.CNF):
            nfe += m.nfe
            m.reset_nfe_ts()
    return nfe


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

        dataloader = self.initialize()
        self.main(dataloader)

    def initialize(self):
        if self.cfg.dataset == "retail":
            dataset_cls = discrete_datasets.Retail
        elif self.cfg.dataset == "accidents":
            dataset_cls = discrete_datasets.Accidents
        else:
            raise ValueError(f"Unknown dataset {self.cfg.dataset}")

        train_ds = dataset_cls(root=self.file_dir, split="train")
        val_ds = dataset_cls(
            root=self.file_dir, split="val", raw_dict=train_ds.raw_dict,
        )
        test_ds = dataset_cls(
            root=self.file_dir, split="test", raw_dict=train_ds.raw_dict,
        )

        self.num_classes = len(train_ds.raw_dict)
        self.num_items = train_ds.num_items
        log.info(f"There are {self.num_items} variables with {self.num_classes} values")

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

        if self.cfg.model == "dpp":
            self.model = DeterminantalPointProcesses(self.num_classes)
        else:
            self.model = DiscreteSetsFlow(
                self.cfg.dequantization,
                self.num_classes,
                self.cfg.embedding_dim,
                self.cfg.model,
                self.cfg.num_flows,
                self.cfg.num_layers,
                self.cfg.actfn,
            )

        self.model.to(self.device)
        self.ema_model = utils.ExponentialMovingAverage(self.model)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd
        )
        self.loss_meter = utils.RunningAverageMeter(0.99)

        log.info(self.model)
        log.info(self.optimizer)

        self.earlystop_val_nll = 1e10
        self.earlystop_test_nll = None

        return data_loaders

    def main(self, data_loaders):
        train_loader, val_loader, test_loader = data_loaders
        train_data_generator = utils.inf_generator(train_loader)

        start_time = time.time()
        prev_itr = self.iter - 1
        while self.iter < self.cfg.iterations:

            batch = next(train_data_generator)[0].to(self.device)
            logprob = compute_loglik(self.model, batch)

            fwd_nfe = count_nfe(self.model)

            loss = logprob.mean().mul_(-1)
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e20)

            if torch.isnan(grad_norm):
                raise StupidNaNsException("NaN gradient.")

            self.optimizer.step()

            if self.iter >= self.cfg.iterations // 2:
                self.ema_model.apply()

            self.loss_meter.update(loss.item())

            if self.iter % self.cfg.logfreq == 0:
                bwd_nfe = count_nfe(self.model)
                time_per_itr = (time.time() - start_time) / (self.iter - prev_itr)
                prev_itr = self.iter
                log.info(
                    f"Iter {self.iter}"
                    f" | Time {time_per_itr:.4f}"
                    f" | Train NLL {self.loss_meter.val:.4f}({self.loss_meter.avg:.4f})"
                    f" | GradNorm {grad_norm:.4f}"
                    f" | Fwd NFE {fwd_nfe:d}"
                    f" | Bwd NFE {bwd_nfe:d}"
                )
                self.save()
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
        self.model.eval()

        eval_nll_meter = utils.AverageMeter()

        for batch in tqdm(dataloader):
            batch = batch[0].to(self.device)

            eval_logprob = compute_loglik(
                self.model, batch, num_samples=self.cfg.num_eval_samples,
            )
            eval_loss = eval_logprob.mean().mul_(-1)

            eval_nll_meter.update(eval_loss.item(), batch.shape[0])

        log.info(f"Iter {self.iter}" f" | {split} NLL {eval_nll_meter.avg:.4f})")

        if self.cfg.ema_eval and self.iter >= self.cfg.iterations // 2:
            self.ema_model.swap()
        self.model.train()
        return eval_nll_meter.avg

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)


# Import like this for pickling
from train_itemsets import Workspace as W


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


@hydra.main(config_path="configs", config_name="itemsets")
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
