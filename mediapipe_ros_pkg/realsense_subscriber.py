from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD


class RealsenseSubsctiber(Node):
    def __init__(self, callback_func):
        super().__init__("realsense_subscriber")

        self.subscription = self.create_subscription(
            RGBD, "/camera/camera/rgbd", callback_func, 10
        )
