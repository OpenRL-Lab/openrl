from typing import Any, Optional

from gymnasium import Env

from .multiagent_env import MultiAgentEnv
from .scenarios import load


def make(
    id: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Env:
    # load scenario from script
    scenario = load(id + ".py").Scenario()
    # create world

    world = scenario.make_world(render_mode=render_mode)
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.info,
        render_mode=render_mode,
    )

    return env
