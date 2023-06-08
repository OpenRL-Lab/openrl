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
import cv2
import numpy as np
from gymnasium import spaces

from openrl.envs.wrappers.base_wrapper import (
    BaseObservationWrapper,
    BaseRewardWrapper,
    BaseWrapper,
)


class NoopResetEnv(BaseWrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )  # change for new
        assert noops > 0
        obs = None
        for _ in range(noops):
            # obs, _, done, _ = self.env.step(self.noop_action)
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(BaseObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class ClipRewardEnv(BaseRewardWrapper):
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class EpisodicLifeEnv(BaseWrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, reward, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(BaseWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        self.need_change = False
        if "FIRE" in env.unwrapped.get_action_meanings():
            self.need_change = True

        if self.need_change:
            assert env.unwrapped.get_action_meanings()[1] == "FIRE"
            assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        if self.need_change:
            self.env.reset(**kwargs)
            obs, reward, done, truncated, info = self.env.step(1)
            if done:
                self.env.reset(**kwargs)
            obs, reward, done, truncated, info = self.env.step(2)
            if done:
                self.env.reset(**kwargs)
            return obs, info
        else:
            return self.env.reset(**kwargs)
