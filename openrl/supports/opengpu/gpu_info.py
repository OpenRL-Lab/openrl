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
import json
import os
import subprocess

import requests


def preserve_decimal(a: float, keep_num: int = 2):
    mul = 10 ^ keep_num
    return int(a * mul) / float(mul)


class GPUInfo:
    gpu_id: int
    product_name: str
    memory_total: int
    memory_used: int
    memory_free: int
    real_id: int

    def __str__(self):
        if self.real_id == self.gpu_id:
            return "GPU:{} free:\t{}Gb used:{}Gb/{}Gb\t{}".format(
                self.gpu_id,
                preserve_decimal(self.memory_free),
                preserve_decimal(self.memory_used),
                self.memory_total,
                self.product_name,
            )
        else:
            return "GPU:{} real id:{} free:\t{}Gb used:{}Gb/{}Gb\t{}".format(
                self.gpu_id,
                self.real_id,
                preserve_decimal(self.memory_free),
                preserve_decimal(self.memory_used),
                self.memory_total,
                self.product_name,
            )

    def __lt__(self, other):
        if self.memory_free < other.memory_free:
            return True
        if self.memory_free > other.memory_free:
            return False
        if self.memory_free == other.memory_free:
            return self.memory_total < other.memory_total


def get_local_GPU_info():
    cmd = "gpustat --json"

    output = subprocess.getoutput(cmd)
    if "not found" in output:
        print(
            "Can not find gpustat. "
            "Please install gpustat first! "
            "You can install gpustat by 'pip install gpustat'"
        )
        return []

    # Deal with vGPU
    output = output.split("\n")
    new_output = []
    for line in output:
        if "4pdvGPU" not in line:
            new_output.append(line)
    output = "\n".join(new_output)

    if "NVML Shared Library Not Found" in output:
        return []

    gpu_dict = json.loads(output)

    gpus = gpu_dict["gpus"]

    gpu_list = []

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_available_list = list(
            map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        )
        assert (
            len(gpu_available_list) > 0
        ), "Get {} GPUs, should greater than zero!".format(len(gpu_available_list))
    else:
        gpu_available_list = None

    if gpu_available_list:
        gpu_real2id = {}
        for gpu_id, real_id in enumerate(gpu_available_list):
            gpu_real2id[real_id] = gpu_id

    for gpu in gpus:
        gpu_info = GPUInfo()

        gpu_info.real_id = int(gpu["index"])

        if gpu_available_list:
            if gpu_info.real_id not in gpu_available_list:
                continue
            else:
                gpu_info.gpu_id = gpu_real2id[gpu_info.real_id]
        else:
            gpu_info.gpu_id = gpu_info.real_id

        gpu_info.product_name = gpu["name"]
        gpu_info.memory_total = gpu["memory.total"] / 1024.0
        gpu_info.memory_used = gpu["memory.used"] / 1024.0
        gpu_info.memory_free = gpu_info.memory_total - gpu_info.memory_used

        gpu_list.append(gpu_info)
    if gpu_available_list:
        assert len(gpu_available_list) == len(
            gpu_list
        ), 'os.environ["CUDA_VISIBLE_DEVICES"]={}, but get {} GPUs'.format(
            os.environ["CUDA_VISIBLE_DEVICES"], len(gpu_list)
        )
    gpu_list.sort(reverse=True)

    return gpu_list


def get_remote_GPU_info(request_api: str):
    return_result = requests.get(request_api)
    gpu_info_dict = json.loads(return_result.content)
    return gpu_info_dict
