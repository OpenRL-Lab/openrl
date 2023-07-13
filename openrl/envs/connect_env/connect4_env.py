from typing import Tuple

from openrl.envs.connect_env.base_connect_env import BaseConnectEnv


class Connect4Env(BaseConnectEnv):
    def _get_board_size(self) -> Tuple[int, int]:
        return 4, 4

    def _get_num2win(self) -> int:
        return 4


if __name__ == "__main__":
    env = Connect4Env(env_name="connect4")
    obs, info = env.reset()
    obs, reward, done, _, info = env.step(1, is_enemy=True)
    env.close()
