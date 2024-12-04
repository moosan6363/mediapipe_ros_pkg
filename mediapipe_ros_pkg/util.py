import cv2
import numpy as np
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
    quaternion = Quaternion(
        x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
    )
    return quaternion


def quarterion_to_direction_vector(quaternion, reference_vector=np.array([1, 0, 0])):
    quaternion = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
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
