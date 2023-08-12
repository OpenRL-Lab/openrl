from typing import Any, Optional

import dmc2gym
import gymnasium as gym
import numpy as np

# class DmcEnv:
#     def __init__(self):
#         env = dmc2gym.make(
#             domain_name='walker',
#             task_name='walk',
#             seed=42,
#             visualize_reward=False,
#             from_pixels='features',
#             height=224,
#             width=224,
#             frame_skip=2
#         )
#         # self.observation_space = spaces.Box(
#         #     low=np.array([0, 0, 0, 0]),
#         #     high=np.array([self.nrow - 1, self.ncol - 1, self.nrow - 1, self.ncol - 1]),
#         #     dtype=int,
#         # )  # current position and target position
#         # self.action_space = spaces.Discrete(
#         #     5
#         # )


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
):
    env = gym.make(id, render_mode=render_mode)
    # env = dmc2gym.make(
    #         domain_name='walker',
    #         task_name='walk',
    #         seed=42,
    #         visualize_reward=False,
    #         from_pixels='features',
    #         height=224,
    #         width=224,
    #         frame_skip=2
    #     )
    return env
