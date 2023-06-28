from typing import Any, Optional

from gymnasium import Env

from .daily_dialog_env import DailyDialogEnv
from .fake_dialog_env import FakeDialogEnv


def make(
    id: str,
    render_mode: Optional[str] = None,
    cfg: Any = None,
    **kwargs: Any,
) -> Env:
    if id == "daily_dialog":
        env = DailyDialogEnv(cfg=cfg)
    elif id == "fake_dialog_data":
        env = FakeDialogEnv(cfg=cfg)
    else:
        raise NotImplementedError

    return env
