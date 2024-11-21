import os
from glob import glob

from setuptools import find_packages, setup

package_name = "mediapipe_ros_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            os.path.join("share", "ament_index", "resource_index", "packages"),
            [os.path.join("resource", package_name)],
        ),
        (os.path.join("share", package_name), ["package.xml"]),
        (
            os.path.join("share", package_name),
            glob(os.path.join("launch", "*_launch.*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Naoki Nomura",
    maintainer_email="naoki.nomura1221@gmail.com",
    description="Mediapipe ROS2 Project.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mediapipe_gesture_publisher = mediapipe_ros_pkg.mediapipe_gesture_publisher:main",
            "mediapipe_objectron_publisher = mediapipe_ros_pkg.mediapipe_objectron_publisher:main",
            "mediapipe_head_direction_publisher = mediapipe_ros_pkg.mediapipe_head_direction_publisher:main",
            "pointed_object_probability_publisher = mediapipe_ros_pkg.pointed_object_probability_publisher:main",
        ],
    },
)
