from typing import Any, Dict, List, Union

import numpy as np


class BaseReward(object):
    def __init__(self):
        self.step_reward_fn = []
        self.inner_reward_fn = []
        self.batch_reward_fn = []

    def get_reward(self, data: Dict[str, Any]):
        return data["reward"], dict()

    def batch_rewards(self, buffer: Any) -> None:
        pass
