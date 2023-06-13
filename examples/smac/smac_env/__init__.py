import copy
from typing import Callable, List, Optional, Union


from gymnasium import Env

from openrl.envs.common import build_envs
from openrl.configs.config import create_config_parser

from .smac_env import SMACEnv


def smac_make(id, render_mode, disable_env_checker, **kwargs):
    cfg_parser = create_config_parser()
    cfg_parser.add_argument(
        "--map_name", type=str, default=id, help="Which smac map to run on"
    )
    cfg_parser.add_argument("--add_move_state", action="store_true", default=False)
    cfg_parser.add_argument("--add_local_obs", action="store_true", default=False)
    cfg_parser.add_argument("--add_distance_state", action="store_true", default=False)
    cfg_parser.add_argument(
        "--add_enemy_action_state", action="store_true", default=False
    )
    cfg_parser.add_argument("--add_agent_id", action="store_true", default=False)
    cfg_parser.add_argument("--add_visible_state", action="store_true", default=False)
    cfg_parser.add_argument("--add_xy_state", action="store_true", default=False)
    cfg_parser.add_argument("--use_state_agent", action="store_false", default=True)
    cfg_parser.add_argument("--use_mustalive", action="store_false", default=True)
    cfg_parser.add_argument("--add_center_xy", action="store_false", default=True)
    cfg_parser.add_argument("--use_zerohidden", action="store_true", default=False)

    cfg = cfg_parser.parse_args([])

    env = SMACEnv(cfg=cfg)
    return env


def make_smac_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_fns = build_envs(
        make=smac_make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
