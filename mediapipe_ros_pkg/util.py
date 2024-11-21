import functools
import inspect
from typing import Type, TypeVar, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

# === geometry =========================================================================================================


def direction_vector_to_quaternion(
    direction_vector, reference_vector=np.array([1, 0, 0])
):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    rotation_axis = np.cross(reference_vector, direction_vector)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    rotation_angle = np.arccos(np.dot(reference_vector, direction_vector))

    quaternion = R.from_rotvec(rotation_axis * rotation_angle).as_quat()
    return quaternion


def quarterion_to_direction_vector(quaternion, reference_vector=np.array([1, 0, 0])):
    rotation = R.from_quat(quaternion)
    return rotation.apply(reference_vector)


# === partial ==========================================================================================================
def default():
    raise ValueError("This is a dummy function and not meant to be called.")


T = TypeVar("T")  # helps with type inference in some editors


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
    """Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
    Then, the outer `partial` will insert the appropriate default value as the function."""

    if func is not default:
        for k, v in kwargs.items():
            if isinstance(v, functools.partial) and v.func is default:
                kwargs[k] = partial(
                    inspect.signature(func).parameters[k].default, *v.args, **v.keywords
                )
    return functools.partial(func, *args, **kwargs)
