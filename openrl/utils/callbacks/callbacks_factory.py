from typing import Any, Dict, List, Type, Union

from openrl.utils.callbacks.callbacks import BaseCallback, CallbackList
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
from openrl.utils.callbacks.eval_callback import EvalCallback

callbacks_dict = {
    "CheckpointCallback": CheckpointCallback,
    "EvalCallback": EvalCallback,
}


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        callbacks: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> BaseCallback:
        if isinstance(callbacks, dict):
            callbacks = [callbacks]
        callbacks_list = []
        for callback in callbacks:
            if callback["id"] not in callbacks_dict:
                raise ValueError(f"Callback {callback['id']} not found")
            callbacks_list.append(callbacks_dict[callback["id"]](**callback["args"]))
        return CallbackList(callbacks_list)

    @staticmethod
    def register(
        id: str,
        callback_class: Type[BaseCallback],
    ):
        callbacks_dict[id] = callback_class
