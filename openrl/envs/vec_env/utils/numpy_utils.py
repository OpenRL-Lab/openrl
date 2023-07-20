# source code from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/utils/numpy_utils.py
"""Numpy utility functions: concatenate space samples and create empty array."""
from collections import OrderedDict
from functools import singledispatch
from typing import Callable, Iterable, Iterator, List, Optional, Union

import numpy as np
from gymnasium.error import CustomSpaceError
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Space,
    Tuple,
)

__all__ = [
    "concatenate",
    "create_empty_array",
    "iterate_action",
    "single_random_action",
]


@singledispatch
def iterate_action(space: Space, actions) -> Iterator:
    """Iterate over the elements of a batched actions.

    Args:
        space: Space to which `actions` belong to.
        actions: actions to be iterated over.

    Returns:
        Iterator over the elements in `actions`.

    Raises:
        ValueError: Space is not an instance of :class:`gym.Space`

    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@iterate_action.register(Discrete)
def _iterate_discrete(space, actions):
    action = np.squeeze(actions, -1)
    return iter(action)


@iterate_action.register(Box)
@iterate_action.register(MultiDiscrete)
@iterate_action.register(MultiBinary)
def _iterate_base(space, actions):
    return iter(actions)


@iterate_action.register(Tuple)
def _iterate_tuple(space, actions):
    raise NotImplementedError("Not implemented yet.")


@iterate_action.register(Dict)
def _iterate_dict(space, actions):
    raise NotImplementedError("Not implemented yet.")


@iterate_action.register(Space)
def _iterate_custom(space, actions):
    raise CustomSpaceError(
        f"Unable to iterate over {actions}, since {space} "
        "is a custom `gymnasium.Space` instance (i.e. not one of "
        "`Box`, `Dict`, etc...)."
    )


@singledispatch
def concatenate(
    space: Space, items: Iterable, out: Union[tuple, dict, np.ndarray]
) -> Union[tuple, dict, np.ndarray]:
    """Concatenate multiple samples from space into a single object.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        items: Samples to be concatenated.
        out: The output object. This object is a (possibly nested) numpy array.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space is not a valid :class:`gym.Space` instance

    Example:
        >>> from gymnasium.spaces import Box
        >>> import numpy as np
        >>> space = Box(low=0, high=1, shape=(3,), seed=42, dtype=np.float32)
        >>> out = np.zeros((2, 3), dtype=np.float32)
        >>> items = [space.sample() for _ in range(2)]
        >>> concatenate(space, items, out)
        array([[0.77395606, 0.43887845, 0.85859793],
               [0.697368  , 0.09417735, 0.97562236]], dtype=float32)
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_base(space, items, out):
    return np.stack(items, axis=0, out=out)


@concatenate.register(Tuple)
def _concatenate_tuple(space, items, out):
    return tuple(
        concatenate(subspace, [item[i] for item in items], out[i])
        for (i, subspace) in enumerate(space.spaces)
    )


@concatenate.register(Dict)
def _concatenate_dict(space, items, out):
    return OrderedDict(
        [
            (key, concatenate(subspace, [item[key] for item in items], out[key]))
            for (key, subspace) in space.spaces.items()
        ]
    )


@concatenate.register(Space)
def _concatenate_custom(space, items, out):
    return tuple(items)


@singledispatch
def create_empty_array(
    space: Space, n: int = 1, agent_num=1, fn: Callable[..., np.ndarray] = np.zeros
) -> Union[tuple, dict, np.ndarray]:
    """Create an empty (possibly nested) numpy array.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment. If `None`, creates an empty sample from `space`.
        fn: Function to apply when creating the empty numpy array. Examples of such functions are `np.empty` or `np.zeros`.

    Returns:
        The output object. This object is a (possibly nested) numpy array.

    Raises:
        ValueError: Space is not a valid :class:`gym.Space` instance

    Example:
        >>> from gymnasium.spaces import Box, Dict
        >>> import numpy as np
        >>> space = Dict({
        ... 'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ... 'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)})
        >>> create_empty_array(space, n=2, fn=np.zeros)
        OrderedDict([('position', array([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)), ('velocity', array([[0., 0.],
               [0., 0.]], dtype=float32))])
    """
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


# It is possible for the some of the Box low to be greater than 0, then array is not in space
@create_empty_array.register(Box)
@create_empty_array.register(MultiDiscrete)
@create_empty_array.register(MultiBinary)
def _create_empty_array_base(space, n=1, agent_num=1, fn=np.zeros):
    # shape = space.shape if (agent_num is None) else (agent_num,) + space.shape
    shape = (agent_num,) + space.shape
    shape = shape if (n is None) else (n,) + shape
    return fn(shape, dtype=space.dtype)


# If the Discrete start > 0 or start + length < 0 then array is not in space
@create_empty_array.register(Discrete)
def _create_empty_array_discrete(space, n=1, agent_num=1, fn=np.zeros):
    # shape = space.shape if (agent_num is None) else (agent_num,) + space.shape
    shape = (agent_num, space.n)
    shape = shape if (n is None) else (n,) + shape
    return fn(shape, dtype=space.dtype)


@create_empty_array.register(Tuple)
def _create_empty_array_tuple(space, n=1, agent_num=1, fn=np.zeros):
    return tuple(
        create_empty_array(subspace, n=n, agent_num=agent_num, fn=fn)
        for subspace in space.spaces
    )


@create_empty_array.register(Dict)
def _create_empty_array_dict(space, n=1, agent_num=1, fn=np.zeros):
    return OrderedDict(
        [
            (key, create_empty_array(subspace, n=n, agent_num=agent_num, fn=fn))
            for (key, subspace) in space.spaces.items()
        ]
    )


@create_empty_array.register(Space)
def _create_empty_array_custom(space, n=1, agent_num=1, fn=np.zeros):
    return None


@singledispatch
def single_random_action(
    space: Space, action_mask: Optional[Union[List[int], np.ndarray]] = None
) -> Union[tuple, dict, np.ndarray]:
    raise ValueError(
        f"Space of type `{type(space)}` is not a valid `gymnasium.Space` instance."
    )


@single_random_action.register(Discrete)
def _single_random_action_discrete(
    space, action_mask: Optional[Union[List[int], np.ndarray]] = None
):
    if action_mask is not None:
        print(action_mask)

        if isinstance(action_mask, list):
            action_mask = np.array(action_mask, dtype=np.int8)
    return [space.sample(mask=action_mask)]


@single_random_action.register(Box)
def _single_random_action_box(
    space, action_mask: Optional[Union[List[int], np.ndarray]] = None
):
    return space.sample()
