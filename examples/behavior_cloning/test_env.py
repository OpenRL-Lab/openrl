""""""
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make


def test_env():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # create environment, set environment parallelism to 9
    env = make("OfflineEnv", env_num=1, cfg=cfg, asynchronous=True)

    for ep_index in range(10):
        done = False
        step = 0
        env.reset()
        while not np.all(done):
            obs, reward, done, info = env.step(env.random_action())
            step += 1
        print(ep_index, step)


if __name__ == "__main__":
    test_env()
