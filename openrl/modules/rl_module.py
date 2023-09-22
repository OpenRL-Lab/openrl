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


def get_train_ds_config(offload,
                        use_fp16=False,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 8,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": use_fp16,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


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

        self.use_deepspeed = cfg.use_deepspeed

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

            if self.program_type == "actor":
                continue

            if not self.use_deepspeed:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=model_cg["lr"],
                    eps=cfg.opti_eps,
                    weight_decay=cfg.weight_decay,
                )
                self.models.update({model_key: model})
                self.optimizers.update({model_key: optimizer})

            if self.use_deepspeed:
                import deepspeed
                from deepspeed.ops.adam import FusedAdam
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                from transformers import get_constant_schedule
                
                self.offload = False
                ds_config = get_train_ds_config(
                    offload=self.offload,
                    use_fp16=cfg.use_fp16,
                )
                ds_config['train_micro_batch_size_per_gpu'] = 200
                ds_config['train_batch_size'] = 1600

                AdamOptimizer = DeepSpeedCPUAdam if self.offload else FusedAdam
                optim_params = get_optimizer_grouped_parameters(model, cfg.weight_decay)
                optim = AdamOptimizer(
                    optim_params,
                    lr=model_cg["lr"],
                    betas=(0.9, 0.95)
                )
                
                # LR Scheduler
                lr_scheduler = get_constant_schedule(
                    optimizer=optim,
                )
                
                engine, *_ = \
                    deepspeed.initialize(
                        model=model,
                        optimizer=optim,
                        lr_scheduler=lr_scheduler,
                        config=ds_config
                    )
                self.models.update({model_key: engine})
                self.optimizers.update({model_key: engine})

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
