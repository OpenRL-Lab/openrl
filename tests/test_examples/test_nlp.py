# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Copyright 2023 The OpenRL Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# """"""
#


from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def config():
    from openrl.configs.config import create_config_parser

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    return cfg


def test_train_nlp(config):
    env = make("fake_dialog_data", env_num=3, cfg=config)
    agent = Agent(Net(env))
    agent.train(total_time_steps=1000)


if __name__ == "__main__":
    cfg = config()
    test_train_nlp(cfg)
