import os
import sys

import numpy as np
import pytest
import ray
import torch

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.envs.wrappers import FlattenObservation
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.selfplay.wrappers.opponent_pool_wrapper import OpponentPoolWrapper
from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper


@pytest.fixture(
    scope="module",
    params=[
        {"port": 13486, "strategy": "RandomOpponent"},
        {"port": 13487, "strategy": "LastOpponent"},
    ],
)
def config(request):
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "./examples/selfplay/selfplay.yaml"])
    cfg.selfplay_api.port = request.param["port"]
    for i, c in enumerate(cfg.callbacks):
        if c["id"] == "SelfplayCallback":
            c["args"][
                "opponent_template"
            ] = "./examples/selfplay/opponent_templates/tictactoe_opponent"
            port = c["args"]["api_address"].split(":")[-1].split("/")[0]
            c["args"]["api_address"] = c["args"]["api_address"].replace(
                port, str(request.param["port"])
            )
            cfg.callbacks[i] = c
        elif c["id"] == "SelfplayAPI":
            c["args"]["sample_strategy"] = request.param["strategy"]
            c["args"]["port"] = request.param["port"]
            cfg.callbacks[i] = c

        else:
            pass

    return cfg


def train(cfg):
    # Create environment
    env_num = 2
    render_model = None
    env = make(
        "tictactoe_v3",
        render_mode=render_model,
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RecordWinner, OpponentPoolWrapper],
        env_wrappers=[FlattenObservation],
        cfg=cfg,
    )
    # Create neural network

    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    # Create agent
    agent = Agent(net)
    # Begin training
    agent.train(total_time_steps=20)
    env.close()
    agent.save("./selfplay_agent/")
    return agent


def evaluation():
    print("Evaluation...")
    env_num = 1
    env = make(
        "tictactoe_v3",
        env_num=env_num,
        asynchronous=True,
        opponent_wrappers=[RandomOpponentWrapper],
        env_wrappers=[FlattenObservation],
        auto_reset=False,
    )

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args([])
    net = Net(env, cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(net)

    agent.load("./selfplay_agent/")
    agent.set_env(env)
    env.reset(seed=0)

    total_reward = 0.0
    ep_num = 2
    for ep_now in range(ep_num):
        obs, info = env.reset()
        done = False
        step = 0

        while not np.any(done):
            # predict next action based on the observation
            action, _ = agent.act(obs, info, deterministic=True)
            obs, r, done, info = env.step(action)
            step += 1

            if np.any(done):
                total_reward += np.mean(r) > 0
                print(f"{ep_now}/{ep_num}: reward: {np.mean(r)}")
    print(f"win rate: {total_reward/ep_num}")
    env.close()
    print("Evaluation finished.")


@pytest.mark.unittest
def test_train_selfplay(config):
    train(config)
    evaluation()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", os.path.basename(__file__)]))
