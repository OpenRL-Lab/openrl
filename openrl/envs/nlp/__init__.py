# MPE env fetched from https://github.com/marlbenchmark/on-policy/tree/main/onpolicy/envs/mpe
from typing import Callable, List, Optional, Union

from gymnasium import Env

from openrl.envs.common import build_envs
from openrl.envs.nlp.nlp_env import make


def make_nlp_envs(
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> List[Callable[[], Env]]:
    from openrl.envs.wrappers import AutoReset  # DictWrapper,
    from openrl.envs.wrappers import RemoveTruncated, Single2MultiAgentWrapper

    env_wrappers = [
        # DictWrapper,
        Single2MultiAgentWrapper,
        AutoReset,
        RemoveTruncated,
    ]
    env_fns = build_envs(
        make=make,
        id=id,
        env_num=env_num,
        render_mode=render_mode,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns
