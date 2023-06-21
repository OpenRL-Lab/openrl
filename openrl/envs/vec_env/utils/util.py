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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# source from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py#L22
def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


def prepare_available_actions(
    info: Optional[List[Dict[str, Any]]] = None,
    agent_num: int = 1,
    as_batch: bool = True,
) -> Optional[np.ndarray]:
    if info is None:
        return None

    available_actions = []
    for env_index in range(len(info)):
        env_info = info[env_index]
        available_actions_env = []
        for agent_index in range(agent_num):
            if env_info is None:
                available_action = None
            else:
                if "available_actions" in env_info:
                    available_action = env_info["available_actions"][agent_index]
                else:
                    # if there is no available_actions in env_info, then we assume all actions are available
                    return None
            available_actions_env.append(available_action)
        available_actions.append(available_actions_env)
    available_actions = np.array(available_actions, dtype=np.int8)
    if as_batch:
        available_actions = available_actions.reshape(-1, available_actions.shape[-1])
    return available_actions
