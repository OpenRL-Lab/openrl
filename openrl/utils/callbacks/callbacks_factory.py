from typing import Dict, Any, Type, List, Union

from openrl.utils.callbacks.callbacks import BaseCallback, CallbackList
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback

callbacks_dict = {
    "CheckpointCallback": CheckpointCallback,
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
            callbacks_list.append(callbacks_dict[callback["id"]](**callback["args"]))
        return CallbackList(callbacks_list)

    @staticmethod
    def register(
        id: str,
        callback_class: Type[BaseCallback],
    ):
        callbacks_dict[id] = callback_class
