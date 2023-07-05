""""""
from openrl.configs.config import create_config_parser
from openrl.envs.common import make

def test_env():
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()

    # We use ZeroRewardWrapper to make sure that we don't get any reward from the environment.
    # create environment, set environment parallelism to 9
    env = make("OfflineEnv", env_num=2, cfg=cfg)

if __name__ == "__main__":
    test_env()

