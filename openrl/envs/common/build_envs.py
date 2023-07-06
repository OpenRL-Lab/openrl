import copy
from typing import Callable, Iterable, List, Optional, Union

from gymnasium import Env


def build_envs(
    make,
    id: str,
    env_num: int = 1,
    render_mode: Optional[Union[str, List[str]]] = None,
    disable_env_checker: Optional[bool] = None,
    wrappers: Optional[Union[Callable[[Env], Env], List[Callable[[Env], Env]]]] = None,
    need_env_id: bool = False,
    **kwargs,
) -> List[Callable[[], Env]]:
    def create_env(env_id: int, env_num: int, need_env_id: bool) -> Callable[[], Env]:
        """Creates an environment that can enable or disable the environment checker."""
        # If the env_id > 0 then disable the environment checker otherwise use the parameter
        _disable_env_checker = True if env_id > 0 else disable_env_checker

        def _make_env() -> Env:
            if isinstance(render_mode, list):
                env_render_mode = render_mode[env_id]
            else:
                env_render_mode = render_mode
            new_kwargs = copy.deepcopy(kwargs)
            if need_env_id:
                new_kwargs["env_id"] = env_id
                new_kwargs["env_num"] = env_num

            env = make(
                id,
                render_mode=env_render_mode,
                disable_env_checker=_disable_env_checker,
                **new_kwargs,
            )

            if wrappers is not None:
                if callable(wrappers):
                    env = wrappers(env)
                elif isinstance(wrappers, Iterable) and all(
                    [callable(w) for w in wrappers]
                ):
                    for wrapper in wrappers:
                        env = wrapper(env)
                else:
                    raise NotImplementedError

            return env

        return _make_env

    env_fns = [create_env(env_id, env_num, need_env_id) for env_id in range(env_num)]
    return env_fns
