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
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union

from openrl.selfplay.callbacks.base_callback import BaseSelfplayCallback
from openrl.selfplay.opponents.opponent_template import OpponentTemplate
from openrl.selfplay.selfplay_api.selfplay_client import SelfPlayClient


class SelfplayCallback(BaseSelfplayCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True`` to save replay buffer checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param opponent_pool_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer

    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        opponent_pool_path: Union[str, Path],
        api_address: str,
        name_prefix: str = "opponent",
        save_replay_buffer: bool = False,
        opponent_template: Optional[str] = None,
        clear_past_opponents: bool = False,
        copy_script_file: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.save_freq = save_freq
        if isinstance(opponent_pool_path, str):
            self.opponent_pool_path = Path(opponent_pool_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.api_address = api_address
        self.api_client = SelfPlayClient(api_address)
        self.opponent_template = OpponentTemplate(opponent_template, copy_script_file)
        self.clear_past_opponents = clear_past_opponents

    def _init_callback(self) -> None:
        if self.clear_past_opponents and self.opponent_pool_path.exists():
            shutil.rmtree(self.opponent_pool_path)
            if self.verbose >= 2:
                print(f"Removed past opponents in {self.opponent_pool_path}")
        # Create folder if needed
        self.last_opponent_link = Path(self.opponent_pool_path) / "latest"
        if self.opponent_pool_path is not None:
            os.makedirs(self.opponent_pool_path, exist_ok=True)
        self.save_opponent()

    def save_opponent(self):
        opponent_path = self.get_opponent_path()
        self.agent.save(opponent_path)

        opponent_info = {"num_time_steps": self.num_time_steps}
        self.opponent_template.save(opponent_path, opponent_info)
        # json.dump(info, open(opponent_path / "info.json", "w"))

        if os.path.islink(self.last_opponent_link):
            os.unlink(self.last_opponent_link)

        os.symlink(opponent_path.absolute(), self.last_opponent_link)

        response = self.api_client.add_opponent(
            opponent_path.stem,
            {
                "opponent_path": str(opponent_path.absolute()),
                "opponent_type": self.opponent_template.opponent_info["opponent_type"],
            },
        )

        if self.verbose >= 2:
            print(response)
            print(f"Opponent is saved to {str(opponent_path.absolute())}")

    def get_opponent_path(self, checkpoint_type: str = "", extension: str = "") -> Path:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return (
            Path(self.opponent_pool_path)
            / "opponents"
            / f"{self.name_prefix}_{checkpoint_type}{self.num_time_steps}_steps{'.' if extension else ''}{extension}"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_opponent()

            if (
                # TODO: add buffer save support
                self.save_replay_buffer
                and hasattr(self.agent, "replay_buffer")
                and self.agent.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path(
                    "replay_buffer_", extension="pkl"
                )
                self.agent.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(
                        f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                    )

        return True
