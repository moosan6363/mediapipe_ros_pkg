from abc import abstractmethod

import numpy as np
from pyrealsense2 import intrinsics, rs2_deproject_pixel_to_point
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD


class RealsenseSubsctiber(Node):
    def __init__(self, node_name, topic_name):
        super().__init__(node_name)

        self.subscription = self.create_subscription(
            RGBD, topic_name, self.callback, 10
        )

    @abstractmethod
    def callback(self, msg):
        pass


def estimate_object_points(
    image_points_dict,
    previous_object_points_dict,
    depth_image,
    rgb_camera_info,
    crop_size_max=10,
    percentile=25,
):
    object_points_dict = {}
    _intrinsics = intrinsics()
    _intrinsics.width = rgb_camera_info.width
    _intrinsics.height = rgb_camera_info.height
    _intrinsics.fx = rgb_camera_info.k[0]
    _intrinsics.fy = rgb_camera_info.k[4]
    _intrinsics.ppx = rgb_camera_info.k[2]
    _intrinsics.ppy = rgb_camera_info.k[5]

    for key, image_point in image_points_dict.items():
        if image_point is None:
            object_points_dict[key] = None
            continue
        else:
            depth_value = previous_object_points_dict[key][2]
            if depth_value > 0:
                # Inverse proportion with depth value
                crop_size = max(1, int(crop_size_max / depth_value))
            else:
                crop_size = 1

            half_crop_size = crop_size // 2
            cropped_depth = depth_image[
                max(0, image_point[1] - half_crop_size) : min(
                    depth_image.shape[0] - 1, image_point[1] + half_crop_size + 1
                ),
                max(0, image_point[0] - half_crop_size) : min(
                    depth_image.shape[1] - 1, image_point[0] + half_crop_size + 1
                ),
            ]

            # Filter out invalid depth values
            valid_depths = cropped_depth[cropped_depth > 0]

            # Crop around the pixel and get the depth value using the nearest 12.5% of the depth value
            # TODO: Fix hard code
            # breakpoint()
            if valid_depths.size > 0:
                depth = np.percentile(valid_depths, percentile)
                point_3d = rs2_deproject_pixel_to_point(
                    _intrinsics, image_point, depth * 0.001
                )
                object_points_dict[key] = point_3d
            else:
                object_points_dict[key] = None

    return object_points_dict
