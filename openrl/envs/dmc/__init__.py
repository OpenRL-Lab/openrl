import copy
from typing import Callable, List, Optional, Union

import dmc2gym

from openrl.envs.common import build_envs
from openrl.envs.dmc.dmc_env import make


def make_dmc_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    from openrl.envs.wrappers import (
        RemoveTruncated,
        Single2MultiAgentWrapper,
    )
    from openrl.envs.wrappers.extra_wrappers import ConvertEmptyBoxWrapper

    env_wrappers = copy.copy(kwargs.pop("env_wrappers", []))
    env_wrappers += [ConvertEmptyBoxWrapper, RemoveTruncated, Single2MultiAgentWrapper]
    env_fns = build_envs(
        make=make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )

    return env_fns
