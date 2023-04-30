#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The TARTRL Authors.
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
import argparse
import logging
import traceback
from typing import List, Union

from openrl.supports.opengpu.gpu_info import get_local_GPU_info, get_remote_GPU_info


class RemoteGPUManager:
    def __init__(self, pytorch_config=None, check: bool = False):
        self.gpu_info_dict = get_remote_GPU_info()
        self.pytorch_config = pytorch_config
        self.server_list = []
        if self.pytorch_config is not None:
            for server_address in self.pytorch_config.GPU_usage_dict:
                self.server_list.append(server_address)

        if check:
            self.check_gpus()

        self.cal_learner_number()

    def check_gpus(self):
        assert self.pytorch_config is not None
        assert len(self.server_list) > 0

        bad_gpus = []
        for server_address in self.server_list:
            assert (
                server_address in self.gpu_info_dict
            ), "can not get gpu info from {}".format(server_address)
            assert len(self.gpu_info_dict[server_address]["gpu_infos"]) > 0

            for gpu_info in self.gpu_info_dict[server_address]["gpu_infos"]:
                if (
                    self.pytorch_config.GPU_usage_dict[server_address]["gpus"] == "all"
                    or gpu_info["gpu"]
                    in self.pytorch_config.GPU_usage_dict[server_address]["gpus"]
                ):
                    if (
                        gpu_info["memory"]["total"] - gpu_info["memory"]["used"]
                        < self.pytorch_config.min_memory_per_gpu
                    ):
                        bad_gpus.append(
                            {
                                "server": server_address,
                                "gpu": gpu_info["gpu"],
                                "free": (
                                    gpu_info["memory"]["total"]
                                    - gpu_info["memory"]["used"]
                                ),
                            }
                        )
        if len(bad_gpus) > 0:
            for bad_gpu in bad_gpus:
                print(
                    "server:{} GPU:{}, minimal memory {}GB, but only get {}GB free"
                    " memory.".format(
                        bad_gpu["server"],
                        bad_gpu["gpu"],
                        self.pytorch_config.min_memory_per_gpu,
                        bad_gpu["free"],
                    )
                )
            assert False, "GPUs not satisfy."

    def cal_learner_number(self):
        self.server_gpu_mapping = {}
        gpu_num = 0
        for server_address in self.server_list:
            gpu_mapping = {}
            for gpu_info in self.gpu_info_dict[server_address]["gpu_infos"]:
                if (
                    self.pytorch_config.GPU_usage_dict[server_address]["gpus"] == "all"
                    or gpu_info["gpu"]
                    in self.pytorch_config.GPU_usage_dict[server_address]["gpus"]
                ):
                    gpu_mapping[gpu_info["gpu"]] = gpu_num
                    gpu_num += 1
            self.server_gpu_mapping[server_address] = gpu_mapping
        self.learner_num = gpu_num

    def get_gpu_info(self, server_list: list):
        gpu_infos = {}
        for server_address in server_list:
            if server_address in self.gpu_info_dict:
                gpu_infos[server_address] = self.gpu_info_dict[server_address]
        return gpu_infos


class LocalGPUManager:
    def __init__(self, args: argparse.Namespace = None):
        self.args = args
        self.gpus = []
        self.learner_num = 0 if args is None else args.learner_num
        if args is None or not self.args.disable_cuda:
            try:
                print("LocalGPUManager fetch gpu infos....")
                self.gpus = get_local_GPU_info()
                print("LocalGPUManager fetch gpu infos done!")
            except Exception:
                print("can not find GPU")

                traceback.print_exc()
                exit()

    def get_gpu(self) -> int:
        if len(self.gpus) == 0:
            return None

        if self.args is None:
            return self.gpus[0].gpu_id

        if self.args.disable_cuda:
            return None
        else:
            return self.gpus[0].gpu_id

    def get_learner_gpu(self, learner_id: int = 0) -> Union[int, None]:
        if self.args is None or (self.args.disable_cuda or len(self.gpus) == 0):
            return None

        if self.args.gpu_usage_type == "auto":
            return self.gpus[
                learner_id if learner_id < len(self.gpus) else len(self.gpus) - 1
            ].gpu_id
        elif self.args.gpu_usage_type == "single":
            return self.gpus[0].gpu_id
        else:
            logging.warning(
                "unknown gpu usage type:{}!".format(self.args.gpu_usage_type)
            )
            return None

    def get_learner_gpus(self) -> List[int]:
        gpus = []
        for learner_id in range(self.learner_num):
            if self.args.gpu_usage_type == "auto":
                gpus.append(
                    self.gpus[
                        (
                            learner_id
                            if learner_id < len(self.gpus)
                            else len(self.gpus) - 1
                        )
                    ].gpu_id
                )
            elif self.args.gpu_usage_type == "single":
                gpus.append(self.gpus[0].gpu_id)
            else:
                logging.warning(
                    "unknown gpu usage type:{}!".format(self.args.gpu_usage_type)
                )
                gpus.append(None)
        return gpus

    def get_worker_gpu(self, worker_id: int = 0) -> int:
        if self.args is None or (self.args.disable_cuda or len(self.gpus) == 0):
            return None

        worker_id += self.args.learner_num

        if self.args.gpu_usage_type == "auto":
            return self.gpus[
                worker_id if worker_id < len(self.gpus) else len(self.gpus) - 1
            ].gpu_id
        elif self.args.gpu_usage_type == "single":
            return self.gpus[0].gpu_id
        else:
            logging.warning(
                "unknown gpu usage type:{}!".format(self.args.gpu_usage_type)
            )
            return None

    def log_info(self):
        if self.args and self.args.disable_cuda:
            return

        for gpu in self.gpus:
            print(gpu)
