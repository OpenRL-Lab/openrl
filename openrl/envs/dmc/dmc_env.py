from typing import Any, Optional

import gymnasium as gym
import numpy as np


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
):
    env = gym.make(id, render_mode=render_mode)
    return env
