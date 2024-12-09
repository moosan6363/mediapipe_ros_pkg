from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_launch_description():
    top_camera_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
        ),
        launch_arguments=[
            ("enable_color", "true"),
            ("enable_depth", "true"),
            ("enable_sync", "true"),
            ("align_depth.enable", "true"),
            ("enable_rgbd", "true"),
            ("camera_name", "top_camera"),
            ("initial_reset", "true"),
            ("global_time_enabled", "false"),
            ("serial_no", "_105322252074"),
        ],
    )
    left_camera_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
        ),
        launch_arguments=[
            ("enable_color", "true"),
            ("enable_depth", "true"),
            ("enable_sync", "true"),
            ("align_depth.enable", "true"),
            ("enable_rgbd", "true"),
            ("camera_name", "left_camera"),
            ("initial_reset", "true"),
            ("global_time_enabled", "false"),
            ("serial_no", "_032522250227"),
        ],
    )

    d455_link_to_color_tvec = [-0.0003, -0.0591, 0.0]
    d455_link_to_color_rmat = R.from_quat([0.0, 0.0, 0.0, 1.0]).as_matrix()
    d455_link_to_color_T = np.eye(4)
    d455_link_to_color_T[:3, :3] = d455_link_to_color_rmat
    d455_link_to_color_T[:3, 3] = d455_link_to_color_tvec

    d455_color_to_color_optical_tvec = [0.0, 0.0, 0.0]
    d455_color_to_color_optical_R = R.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()
    d455_color_to_color_optical_T = np.eye(4)
    d455_color_to_color_optical_T[:3, :3] = d455_color_to_color_optical_R
    d455_color_to_color_optical_T[:3, 3] = d455_color_to_color_optical_tvec

    world_to_top_camera_color_optical_T = np.array(
        [
            [
                -0.7156728506088257,
                -0.49103277921676636,
                0.49668821692466736,
                0.22494880855083466,
            ],
            [
                -0.6984298825263977,
                0.5002667307853699,
                -0.511789858341217,
                0.47752347588539124,
            ],
            [
                0.0028289840556681156,
                -0.7131760120391846,
                -0.7009792923927307,
                1.500551462173462,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world_to_top_camera_link_T = (
        world_to_top_camera_color_optical_T
        @ np.linalg.pinv(d455_color_to_color_optical_T)
        @ np.linalg.pinv(d455_link_to_color_T)
    )

    top_camera_static_tf = Node(
        package="mediapipe_ros_pkg",
        name="static_frame_publisher",
        executable="static_frame_publisher",
        arguments=[
            "world",
            "top_camera_link",
            f"{world_to_top_camera_link_T[0, 0]}",
            f"{world_to_top_camera_link_T[0, 1]}",
            f"{world_to_top_camera_link_T[0, 2]}",
            f"{world_to_top_camera_link_T[0, 3]}",
            f"{world_to_top_camera_link_T[1, 0]}",
            f"{world_to_top_camera_link_T[1, 1]}",
            f"{world_to_top_camera_link_T[1, 2]}",
            f"{world_to_top_camera_link_T[1, 3]}",
            f"{world_to_top_camera_link_T[2, 0]}",
            f"{world_to_top_camera_link_T[2, 1]}",
            f"{world_to_top_camera_link_T[2, 2]}",
            f"{world_to_top_camera_link_T[2, 3]}",
        ],
    )

    world_to_left_camera_color_optical_T = np.array(
        [
            [
                0.8517307639122009,
                -0.31086865067481995,
                0.4218001067638397,
                0.40282782912254333,
            ],
            [
                -0.5238497257232666,
                -0.4872702360153198,
                0.6986767053604126,
                -0.65426105260849,
            ],
            [
                -0.011666045524179935,
                -0.8160443305969238,
                -0.577871561050415,
                0.6526107788085938,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world_to_left_camera_link_T = (
        world_to_left_camera_color_optical_T
        @ np.linalg.pinv(d455_color_to_color_optical_T)
        @ np.linalg.pinv(d455_link_to_color_T)
    )

    left_camera_static_tf = Node(
        package="mediapipe_ros_pkg",
        name="static_frame_publisher",
        executable="static_frame_publisher",
        arguments=[
            "world",
            "left_camera_link",
            f"{world_to_left_camera_link_T[0, 0]}",
            f"{world_to_left_camera_link_T[0, 1]}",
            f"{world_to_left_camera_link_T[0, 2]}",
            f"{world_to_left_camera_link_T[0, 3]}",
            f"{world_to_left_camera_link_T[1, 0]}",
            f"{world_to_left_camera_link_T[1, 1]}",
            f"{world_to_left_camera_link_T[1, 2]}",
            f"{world_to_left_camera_link_T[1, 3]}",
            f"{world_to_left_camera_link_T[2, 0]}",
            f"{world_to_left_camera_link_T[2, 1]}",
            f"{world_to_left_camera_link_T[2, 2]}",
            f"{world_to_left_camera_link_T[2, 3]}",
        ],
    )
    return LaunchDescription(
        [
            top_camera_launch,
            left_camera_launch,
            top_camera_static_tf,
            left_camera_static_tf,
        ]
    )
