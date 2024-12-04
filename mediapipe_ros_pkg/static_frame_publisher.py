import sys

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def main():
    rclpy.init()
    static_frame_publisher = StaticFramePublisher(sys.argv)
    try:
        rclpy.spin(static_frame_publisher)
    except KeyboardInterrupt:
        static_frame_publisher.destroy_node()


class StaticFramePublisher(Node):
    """
    Broadcast transforms that never change.
    """

    def __init__(self, transformation):
        super().__init__("static_frame_broadcaster")

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Publish static transforms once at startup
        self.make_transforms(transformation)

    def make_transforms(self, transformation):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = transformation[1]
        t.child_frame_id = transformation[2]

        t.transform.translation.x = float(transformation[3])
        t.transform.translation.y = float(transformation[4])
        t.transform.translation.z = float(transformation[5])

        rotation = Rotation.from_euler(
            "xyz",
            [
                float(transformation[6]),
                float(transformation[7]),
                float(transformation[8]),
            ],
            degrees=False,
        ).as_quat()
        rotation = Quaternion(
            x=rotation[0],
            y=rotation[1],
            z=rotation[2],
            w=rotation[3],
        )

        t.transform.rotation = rotation

        self.tf_static_broadcaster.sendTransform(t)
