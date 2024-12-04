import functools
import inspect
from typing import Type, TypeVar, Union

import cv2
import numpy as np
import torch
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation

# === geometry =========================================================================================================


def direction_vector_to_quaternion(
    direction_vector, reference_vector=np.array([1, 0, 0])
):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    rotation_axis = np.cross(reference_vector, direction_vector)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    rotation_angle = np.arccos(np.dot(reference_vector, direction_vector))

    quaternion = Rotation.from_rotvec(rotation_axis * rotation_angle).as_quat()
    return quaternion


def quarterion_to_direction_vector(quaternion, reference_vector=np.array([1, 0, 0])):
    rotation = Rotation.from_quat(quaternion)
    return rotation.apply(reference_vector)


def rmat_to_quaternion(rmat):
    return Rotation.from_matrix(rmat).as_quat()


def quaternion_multiply(q0, q1):
    q = Quaternion(
        x=q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
        y=q0.w * q1.y - q0.x * q1.z + q0.y * q1.w + q0.z * q1.x,
        z=q0.w * q1.z + q0.x * q1.y - q0.y * q1.x + q0.z * q1.w,
        w=q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z,
    )
    return q


def quaternion_from_euler(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = Quaternion(
        x=cy * cp * sr - sy * sp * cr,
        y=sy * cp * sr + cy * sp * cr,
        z=sy * cp * cr - cy * sp * sr,
        w=cy * cp * cr + sy * sp * sr,
    )

    return q


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(
            torch.device("cpu")
        )
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(
            torch.device("cuda:%d" % gpu)
        )
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def write_poining_vector(rgb_image, start_image_point, end_image_point, color):
    try:
        cv2.line(
            rgb_image,
            (int(start_image_point[0]), int(start_image_point[1])),
            (int(end_image_point[0]), int(end_image_point[1])),
            color,
            3,
        )
    except (cv2.error, ValueError) as e:
        print(e)


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
