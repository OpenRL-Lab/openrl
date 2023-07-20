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

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from gymnasium.utils import seeding

from openrl.envs.wrappers.base_wrapper import BaseWrapper


class BaseMultiPlayerWrapper(BaseWrapper, ABC):
    """
    Base class for multi-player wrappers.
    """

    _np_random: Optional[np.random.Generator] = None

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, **kwargs):
        """
        Reset the environment.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            The initial observation.
        """
        raise NotImplementedError

    def close(self):
        self.env.close()

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random
