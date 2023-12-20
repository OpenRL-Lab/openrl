import copy
import inspect
from typing import Callable, Iterable, List, Optional, Union

import envpool
from gymnasium import Env

from openrl.envs.vec_env import (
    AsyncVectorEnv,
    RewardWrapper,
    SyncVectorEnv,
    VecMonitorWrapper,
)
from openrl.envs.vec_env.vec_info import VecInfoFactory
from openrl.envs.wrappers.base_wrapper import BaseWrapper
from openrl.rewards import RewardFactory


def build_envs(
    make,
    id: str,
    env_num: int = 1,
    wrappers: Optional[Union[Callable[[Env], Env], List[Callable[[Env], Env]]]] = None,
    need_env_id: bool = False,
    **kwargs,
) -> List[Callable[[], Env]]:
    cfg = kwargs.get("cfg", None)

    def create_env(env_id: int, env_num: int, need_env_id: bool) -> Callable[[], Env]:
        def _make_env() -> Env:
            new_kwargs = copy.deepcopy(kwargs)
            if need_env_id:
                new_kwargs["env_id"] = env_id
                new_kwargs["env_num"] = env_num
            if "envpool" in new_kwargs:
                # for now envpool doesnt support any render mode
                # envpool also doesnt stores the id anywhere
                new_kwargs.pop("envpool")
                env = make(
                    id,
                    **new_kwargs,
                )
                env.unwrapped.spec.id = id

            if wrappers is not None:
                if callable(wrappers):
                    if issubclass(wrappers, BaseWrapper):
                        env = wrappers(env, cfg=cfg)
                    else:
                        env = wrappers(env)
                elif isinstance(wrappers, Iterable) and all(
                    [callable(w) for w in wrappers]
                ):
                    for wrapper in wrappers:
                        if (
                            issubclass(wrapper, BaseWrapper)
                            and "cfg" in inspect.signature(wrapper.__init__).parameters
                        ):
                            env = wrapper(env, cfg=cfg)
                        else:
                            env = wrapper(env)
                else:
                    raise NotImplementedError

            return env

        return _make_env

    env_fns = [create_env(env_id, env_num, need_env_id) for env_id in range(env_num)]
    return env_fns


def make_envpool_envs(
    id: str,
    env_num: int = 1,
    **kwargs,
):
    assert "env_type" in kwargs
    assert kwargs.get("env_type") in ["gym", "dm", "gymnasium"]
    kwargs["envpool"] = True

    if "env_wrappers" in kwargs:
        env_wrappers = kwargs.pop("env_wrappers")
    else:
        env_wrappers = []
    env_fns = build_envs(
        make=envpool.make,
        id=id,
        env_num=env_num,
        wrappers=env_wrappers,
        **kwargs,
    )
    return env_fns


def make(
    id: str,
    env_num: int = 1,
    asynchronous: bool = False,
    add_monitor: bool = True,
    render_mode: Optional[str] = None,
    auto_reset: bool = True,
    **kwargs,
):
    cfg = kwargs.get("cfg", None)
    if id in envpool.registration.list_all_envs():
        env_fns = make_envpool_envs(
            id=id.split(":")[-1],
            env_num=env_num,
            **kwargs,
        )
        if asynchronous:
            env = AsyncVectorEnv(
                env_fns, render_mode=render_mode, auto_reset=auto_reset
            )
        else:
            env = SyncVectorEnv(env_fns, render_mode=render_mode, auto_reset=auto_reset)

        reward_class = cfg.reward_class if cfg else None
        reward_class = RewardFactory.get_reward_class(reward_class, env)

        env = RewardWrapper(env, reward_class)

        if add_monitor:
            vec_info_class = cfg.vec_info_class if cfg else None
            vec_info_class = VecInfoFactory.get_vec_info_class(vec_info_class, env)
            env = VecMonitorWrapper(vec_info_class, env)

        return env
    else:
        raise NotImplementedError(f"env {id} is not supported")
