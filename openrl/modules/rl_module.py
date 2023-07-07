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
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from gym import spaces

from openrl.modules.base_module import BaseModule
from openrl.modules.model_config import ModelTrainConfig


class RLModule(BaseModule):
    def __init__(
        self,
        cfg,
        act_space: spaces.Box,
        rank: int = 0,
        world_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        model_configs: Optional[Dict[str, ModelTrainConfig]] = None,
    ) -> None:
        super(RLModule, self).__init__(cfg)

        if isinstance(device, str):
            device = torch.device(device)

        self.cfg = cfg
        self.device = device
        self.lr = cfg.lr
        self.critic_lr = cfg.critic_lr
        self.opti_eps = cfg.opti_eps
        self.weight_decay = cfg.weight_decay
        self.load_optimizer = cfg.load_optimizer

        self.act_space = act_space

        self.program_type = cfg.program_type
        self.rank = rank
        self.world_size = world_size

        use_half_actor = self.program_type == "actor" and cfg.use_half_actor

        if model_configs is None:
            model_configs = self.get_model_configs(cfg)

        for model_key in model_configs:
            model_cg = model_configs[model_key]
            model = model_cg["model"](
                cfg=cfg,
                input_space=model_cg["input_space"],
                action_space=act_space,
                device=device,
                use_half=use_half_actor,
                extra_args=model_cg["extra_args"] if "extra_args" in model_cg else None,
            )
            self.models.update({model_key: model})

            if self.program_type == "actor":
                continue

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_cg["lr"],
                eps=cfg.opti_eps,
                weight_decay=cfg.weight_decay,
            )
            self.optimizers.update({model_key: optimizer})

            if cfg.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None

    def get_model_configs(self, cfg) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_actions(self):
        raise NotImplementedError

    @abstractmethod
    def get_values(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self):
        raise NotImplementedError

    @abstractmethod
    def act(self):
        raise NotImplementedError

    @abstractmethod
    def get_critic_value_normalizer(self):
        raise NotImplementedError

    def load_policy(self, model_path: str) -> None:
        model_path = Path(model_path)
        assert (
            model_path.exists()
        ), "can not find policy weight file to load: {}".format(model_path)
        state_dict = torch.load(str(model_path), map_location=self.device)
        if "policy" in self.models:
            self.models["policy"].load_state_dict(state_dict)
        else:
            self.models["model"].load_state_dict(state_dict)
        del state_dict

    def restore(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        assert model_dir.exists(), "can not find model directory to restore: {}".format(
            model_dir
        )

        for model_name in self.models:
            state_dict = torch.load(
                str(model_dir) + "/{}.pt".format(model_name), map_location=self.device
            )
            self.models[model_name].load_state_dict(state_dict)
            del state_dict

        if self.load_optimizer:
            if Path(str(model_dir) + "/actor_optimizer.pt").exists():
                for optimizer_name in self.optimizers:
                    state_dict = torch.load(
                        str(model_dir) + "/{}_optimizer.pt".format(optimizer_name),
                        map_location=self.device,
                    )
                    self.optimizers[optimizer_name].load_state_dict(state_dict)
                    del state_dict
            else:
                print("can't find optimizer to restore")
        # TODO
        # optimizer.load_state_dict(resume_state['optimizer'])

    def save(self, save_dir: str) -> None:
        print("\n\n\nenter here")
        pass
