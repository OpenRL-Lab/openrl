#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import logging
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from rich.logging import RichHandler


class Logger:
    def __init__(
        self,
        cfg,
        project_name: str,
        scenario_name: str,
        wandb_entity: str,
        exp_name: Optional[str] = None,
        log_path: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        log_level: int = logging.DEBUG,
        log_to_terminal: bool = True,
    ) -> None:
        # TODO: change these flags to log_backend
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        self.log_level = log_level
        self.log_path = log_path
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.wandb_entity = wandb_entity
        self.log_to_terminal = log_to_terminal
        if exp_name is not None:
            self.exp_name = exp_name
        else:
            self.exp_name = cfg.experiment_name
        self.cfg = cfg
        self._init()

    def _init(self) -> None:
        running_programs = [
            "learner",
            "server_learner",
            "local",
            "whole",
            "local_evaluator",
        ]
        if self.cfg.program_type not in running_programs:
            return None

        if self.log_path is None:
            assert (not self.use_wandb) and (
                not self.use_tensorboard
            ), "log_path must be set when using wandb or tensorboard"
            self.use_wandb = False
            self.use_tensorboard = False
            run_dir = None
        else:
            run_dir = (
                Path(self.log_path)
                / self.project_name
                / self.scenario_name
                / self.exp_name
            )

            if not run_dir.exists():
                os.makedirs(str(run_dir))

            if not self.use_wandb:
                if not run_dir.exists():
                    curr_run = "run1"
                else:
                    exst_run_nums = [
                        int(str(folder.name).split("run")[1])
                        for folder in run_dir.iterdir()
                        if str(folder.name).startswith("run")
                    ]
                    if len(exst_run_nums) == 0:
                        curr_run = "run1"
                    else:
                        curr_run = "run%i" % (max(exst_run_nums) + 1)
                run_dir = run_dir / curr_run
                if not run_dir.exists():
                    os.makedirs(str(run_dir))

        if hasattr(self.cfg, "render"):
            self.cfg.render_save_path = run_dir / "render.png"

        handlers = [RichHandler()]
        if run_dir is not None:
            log_file = os.path.join(run_dir, "log.txt")
            handlers.append(logging.FileHandler(log_file))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=handlers,
        )

        if self.use_wandb:
            wandb.init(
                config=self.cfg,
                project=self.project_name,
                entity=self.wandb_entity,
                notes=socket.gethostname(),
                name=self.scenario_name
                + "_"
                + str(self.exp_name)
                + "_seed"
                + str(self.cfg.seed),
                dir=str(run_dir),
                job_type="training",
                reinit=True,
            )
        elif self.use_tensorboard:
            from tensorboardX import SummaryWriter

            self.log_dir = str(run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)

        self.run_dir = run_dir

    def close(self):
        if self.use_wandb:
            wandb.finish()

    def info(self, msg: str):
        logging.info(msg)

    def log_learner_info(
        self,
        leaner_id: int,
        infos: Dict[str, Any],
        step: int,
    ) -> None:
        if not (self.use_wandb or self.use_tensorboard):
            return
        for k, v in infos.items():
            if self.use_wandb:
                wandb.log({"Learner_{}/{}".format(leaner_id, k): v}, step=step)
            elif self.use_tensorboard:
                self.writter.add_scalars(
                    "Learner_{}/{}".format(leaner_id, k),
                    {"Learner_{}/{}".format(leaner_id, k): v},
                    step,
                )

    def log_info(
        self,
        infos: Dict[str, Any],
        step: int,
    ) -> None:
        if not (self.use_wandb or self.use_tensorboard or self.log_to_terminal):
            return
        logging_info_str = "\n"
        for k, v in infos.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if not isinstance(v, (int, float)):
                v = np.mean(v)
            logging_info_str += f"\t{k}: {v}\n"

            if self.use_wandb:
                wandb.log({k: v}, step=step)
            elif self.use_tensorboard:
                self.writter.add_scalars(k, {k: v}, step)
        if self.log_to_terminal:
            logging.info(logging_info_str)
