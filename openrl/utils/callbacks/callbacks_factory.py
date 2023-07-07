from typing import Any, Dict, List, Type, Union

from openrl.utils.callbacks.callbacks import BaseCallback, CallbackList, EveryNTimesteps
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
from openrl.utils.callbacks.eval_callback import EvalCallback
from openrl.utils.callbacks.processbar_callback import ProgressBarCallback
from openrl.utils.callbacks.stop_callback import (
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)

callbacks_dict = {
    "CheckpointCallback": CheckpointCallback,
    "EvalCallback": EvalCallback,
    "StopTrainingOnRewardThreshold": StopTrainingOnRewardThreshold,
    "StopTrainingOnMaxEpisodes": StopTrainingOnMaxEpisodes,
    "StopTrainingOnNoModelImprovement": StopTrainingOnNoModelImprovement,
    "ProgressBarCallback": ProgressBarCallback,
    "EveryNTimesteps": EveryNTimesteps,
}


class CallbackFactory:
    @staticmethod
    def get_callback(
        callback: Dict[str, Any],
    ) -> BaseCallback:
        if callback["id"] not in callbacks_dict:
            raise ValueError(f"Callback {callback['id']} not found")
        if "args" in callback:
            callback = callbacks_dict[callback["id"]](**callback["args"])
        else:
            callback = callbacks_dict[callback["id"]]()
        return callback

    @staticmethod
    def get_callbacks(
        callbacks: Union[Dict[str, Any], List[Dict[str, Any]]],
        stop_logic: str = "OR",
    ) -> CallbackList:
        if isinstance(callbacks, dict):
            callbacks = [callbacks]
        callbacks_list = []
        for callback in callbacks:
            if callback["id"] not in callbacks_dict:
                raise ValueError(f"Callback {callback['id']} not found")
            callbacks_list.append(CallbackFactory.get_callback(callback))
        return CallbackList(callbacks_list, stop_logic=stop_logic)

    @staticmethod
    def register(
        id: str,
        callback_class: Type[BaseCallback],
    ):
        callbacks_dict[id] = callback_class
