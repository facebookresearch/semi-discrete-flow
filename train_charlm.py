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

setproctitle("train_charlm")

from omegaconf import OmegaConf
import hydra

import time
import math
import logging
import os
import random
import numpy as np

import pickle as pkl

import json
import wandb

import torch
import torch.optim as optim

from survae.utils import elbo_bpd, iwbo_bpd

from charlm.data.data import get_data
from charlm.model.model import get_model
from charlm.optim.expdecay import get_optim
from charlm.experiment.utils import get_metric_table

import utils

log = logging.getLogger(__name__)


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = os.getcwd()
        self.file_dir = os.path.dirname(__file__)
        log.info(f"workspace: {self.work_dir}")

        args = getattr(getattr(self.cfg, self.cfg.dataset), self.cfg.model)

        # Intervals
        self.eval_every = args.eval_every
        self.check_every = args.check_every

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []
        self.best_eval_bpc = float("inf")

    def run(self):

        try:
            assert "exp/" in self.work_dir
            if os.path.isfile(os.path.join(self.file_dir, ".wandb.key.json")):
                with open(os.path.join(self.file_dir, ".wandb.key.json")) as f:
                    data = json.load(f)
                wandb_apikey = data.get("wandbapikey")
                os.environ["WANDB_API_KEY"] = wandb_apikey
                wandb_name = os.path.relpath(
                    self.work_dir,
                    os.path.join(self.file_dir, f"exp/charlm/{self.cfg.dataset}"),
                )
                wandb.init(
                    project=f"vq-charlm-{self.cfg.dataset}",
                    entity="rtqichen",
                    name=wandb_name,
                    resume="allow",
                )
                log.info(f"wandb initialized for {wandb_name}")
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
        else:
            log.info("WARNING: Using device {}".format(device))

        self.device = device
        self.use_gpu = device.type == "cuda"

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.backends.cudnn.benchmark = True

        try:
            train_loader, valid_loader, test_loader = self.initialize()
            self.main(train_loader, valid_loader, test_loader)
        finally:
            if self.wandb_available:
                wandb.finish()

    def initialize(self):
        args = getattr(getattr(self.cfg, self.cfg.dataset), self.cfg.model)

        train_loader, valid_loader, test_loader, data_shape, num_classes = get_data(
            args
        )

        if self.current_epoch > 0:
            return train_loader, valid_loader, test_loader

        model = get_model(
            self.cfg, args, data_shape=data_shape, num_classes=num_classes
        )
        log.info(model)

        model = model.to(self.device)

        optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)

        self.model = model
        self.ema = utils.ExponentialMovingAverage(model)
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch
        self.epochs = args.epochs

        return train_loader, valid_loader, test_loader

    def main(self, train_loader, valid_loader, test_loader):
        args = getattr(getattr(self.cfg, self.cfg.dataset), self.cfg.model)

        for epoch in range(self.current_epoch, self.epochs):

            # Train
            train_dict = self.train_fn(epoch, train_loader, args)
            self.log_train_metrics(train_dict)
            if self.wandb_available:
                wandb.log({"Epoch": epoch + 1, **train_dict})

            # Eval
            if (epoch + 1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch, valid_loader, test_loader)
                self.log_eval_metrics(eval_dict)
                if self.wandb_available:
                    wandb.log({"Epoch": epoch + 1, **eval_dict})
                self.eval_epochs.append(epoch)
            else:
                eval_dict = None

            # Log
            self.save_metrics()

            # Checkpoint
            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.save("latest")
                if eval_dict["test_bpc"] < self.best_eval_bpc:
                    self.save("best")

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def train_fn(self, epoch, train_loader, args):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        num_consecutive_nan = 0
        prev_itr = 0
        start_time = time.time()
        for iter, (x, _) in enumerate(train_loader):

            self.optimizer.zero_grad()

            loss = elbo_bpd(self.model, x.to(self.device))
            if torch.isnan(loss):
                log.warn("Found NaN loss.")
            else:
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e10)
            if torch.isnan(grad_norm):
                log.warn("Found NaN gradient.")
                num_consecutive_nan += 1
                if num_consecutive_nan > 20:
                    log.error("Consecutive NaN gradients.")
            else:
                self.optimizer.step()
                if self.scheduler_iter:
                    self.scheduler_iter.step()
                if epoch >= args.disable_ema_epochs:
                    self.ema.apply()
                num_consecutive_nan = 0

            if iter % self.cfg.logfreq == 0 or iter == len(train_loader) - 1:
                time_per_itr = (time.time() - start_time) / (iter + 1 - prev_itr)
                prev_itr = iter
                log.info(
                    "Training. Epoch: {}/{}, Datapoint: {}/{}, Time: {:.3f}, Bits/char: {:.3f}".format(
                        epoch + 1,
                        self.epochs,
                        loss_count,
                        len(train_loader.dataset),
                        time_per_itr,
                        loss_sum / loss_count,
                    )
                )
                start_time = time.time()
        log.info("")
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        return {"bpc": loss_sum / loss_count}

    def eval_fn(self, epoch, valid_loader, test_loader):
        args = getattr(getattr(self.cfg, self.cfg.dataset), self.cfg.model)
        self.model.eval()
        if epoch > args.disable_ema_epochs:
            self.ema.swap()
        with torch.no_grad():
            valid_loss_sum = 0.0
            valid_loss_count = 0
            for iter, (x, _) in enumerate(valid_loader):
                loss = iwbo_bpd(self.model, x.to(self.device), k=1)
                valid_loss_sum += loss.detach().cpu().item() * len(x)
                valid_loss_count += len(x)
                if iter % self.cfg.logfreq == 0 or iter == len(valid_loader) - 1:
                    log.info(
                        "Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}".format(
                            epoch + 1,
                            self.epochs,
                            valid_loss_count,
                            len(valid_loader.dataset),
                            valid_loss_sum / valid_loss_count,
                        )
                    )

            log.info("")

            test_loss_sum = 0.0
            test_loss_count = 0
            for iter, (x, _) in enumerate(test_loader):
                loss = elbo_bpd(self.model, x.to(self.device))
                test_loss_sum += loss.detach().cpu().item() * len(x)
                test_loss_count += len(x)
                if iter % self.cfg.logfreq == 0 or iter == len(test_loader) - 1:
                    log.info(
                        "Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}".format(
                            epoch + 1,
                            self.epochs,
                            test_loss_count,
                            len(test_loader.dataset),
                            test_loss_sum / test_loss_count,
                        )
                    )

            log.info("")
        if epoch > args.disable_ema_epochs:
            self.ema.swap()

        return {
            "valid_bpc": valid_loss_sum / valid_loss_count,
            "test_bpc": test_loss_sum / test_loss_count,
        }

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics) == 0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics) == 0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def save_metrics(self):

        # Save metrics
        with open(os.path.join(self.work_dir, "metrics_train.pkl"), "wb") as f:
            pkl.dump(self.train_metrics, f)
        with open(os.path.join(self.work_dir, "metrics_eval.pkl"), "wb") as f:
            pkl.dump(self.eval_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(
            self.train_metrics, epochs=list(range(1, self.current_epoch + 2))
        )
        with open(os.path.join(self.work_dir, "metrics_train.txt"), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(
            self.eval_metrics, epochs=[e + 1 for e in self.eval_epochs]
        )
        with open(os.path.join(self.work_dir, "metrics_eval.txt"), "w") as f:
            f.write(str(metric_table))


# Import like this for pickling
from train_charlm import Workspace as W


@hydra.main(config_path="configs", config_name="charlm")
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
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
