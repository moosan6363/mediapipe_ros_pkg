import numpy as np
from scipy.spatial.transform import Rotation as R


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
