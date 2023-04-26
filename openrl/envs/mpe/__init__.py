# MPE env fetched from https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs/mpe
from typing import Callable, List, Optional, Union

from gymnasium import Env

from openrl.envs.common import build_envs
from openrl.envs.mpe.mpe_env import make


def make_mpe_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    env_wrappers = []
    env_fns = build_envs(
        make=make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )

    return env_fns
