from setuptools import find_packages, setup

package_name = "mediapipe_ros_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
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
        ],
    },
)
