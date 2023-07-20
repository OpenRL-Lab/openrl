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

from __future__ import annotations

import operator as op
import typing
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Any, TypeVar, Union, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    GraphInstance,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Space,
    Text,
    Tuple,
)
from numpy.typing import NDArray

T = TypeVar("T")
FlatType = Union[
    NDArray[Any], typing.Dict[str, Any], typing.Tuple[Any, ...], GraphInstance
]


@singledispatch
def flatten(space: Space[T], agent_num: int, x: T) -> FlatType:
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Args:
        space: The space that ``x`` is flattened by
        x: The value to flatten

    Returns:
        The flattened datapoint

            - For :class:`gymnasium.spaces.Box` and :class:`gymnasium.spaces.MultiBinary`, this is a flattened array
            - For :class:`gymnasium.spaces.Discrete` and :class:`gymnasium.spaces.MultiDiscrete`, this is a flattened one-hot array of the sample
            - For :class:`gymnasium.spaces.Tuple` and :class:`gymnasium.spaces.Dict`, this is a concatenated array the subspaces (does not support graph subspaces)
            - For graph spaces, returns :class:`GraphInstance` where:
                - :attr:`GraphInstance.nodes` are n x k arrays
                - :attr:`GraphInstance.edges` are either:
                    - m x k arrays
                    - None
                - :attr:`GraphInstance.edge_links` are either:
                    - m x 2 arrays
                    - None

    Raises:
        NotImplementedError: If the space is not defined in :mod:`gymnasium.spaces`.

    Example:
        >>> from gymnasium.spaces import Box, Discrete, Tuple
        >>> space = Box(0, 1, shape=(3, 5))
        >>> flatten(space, space.sample()).shape
        (15,)
        >>> space = Discrete(4)
        >>> flatten(space, 2)
        array([0, 0, 1, 0])
        >>> space = Tuple((Box(0, 1, shape=(2,)), Box(0, 1, shape=(3,)), Discrete(3)))
        >>> example = ((.5, .25), (1., 0., .2), 1)
        >>> flatten(space, example)
        array([0.5 , 0.25, 1.  , 0.  , 0.2 , 0.  , 1.  , 0.  ])
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten.register(Box)
@flatten.register(MultiBinary)
def _flatten_box_multibinary(
    space: Box | MultiBinary, agent_num: int, x: NDArray[Any]
) -> NDArray[Any]:
    return np.asarray(x, dtype=space.dtype).reshape(agent_num, -1)
