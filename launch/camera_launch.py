from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    front_camera_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
        ),
        launch_arguments=[
            ("enable_color", "true"),
            ("enable_depth", "true"),
            ("enable_sync", "true"),
            ("align_depth.enable", "true"),
            ("enable_rgbd", "true"),
            ("camera_name", "front_camera"),
            ("initial_reset", "true"),
            ("global_time_enabled", "false"),
            ("serial_no", "_213522251272"),
        ],
    )
    side_camera_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
        ),
        launch_arguments=[
            ("enable_color", "true"),
            ("enable_depth", "true"),
            ("enable_sync", "true"),
            ("align_depth.enable", "true"),
            ("enable_rgbd", "true"),
            ("camera_name", "side_camera"),
            ("initial_reset", "true"),
            ("global_time_enabled", "false"),
            ("serial_no", "_234222303418"),
        ],
    )
    side_camera_static_tf = Node(
        package="mediapipe_ros_pkg",
        name="static_frame_publisher",
        executable="static_frame_publisher",
        arguments=[
            "front_camera_link",
            "side_camera_link",
            "0.0",
            "0.22",
            f"{0.59-0.125}",
            "0.0",
            "0.0",
            "0.0",
        ],
    )
    return LaunchDescription(
        [
            front_camera_launch,
            side_camera_launch,
            side_camera_static_tf,
        ]
    )
