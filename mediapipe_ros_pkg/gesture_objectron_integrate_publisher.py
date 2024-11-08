import message_filters
import rclpy
from geometry_msgs.msg import PolygonStamped, PoseArray
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD


def main(args=None):
    rclpy.init(args=args)

    gesture_objectron_integrator = GestureObjectronIntegrator()

    realsense_subscriber = message_filters.Subscriber(
        gesture_objectron_integrator, RGBD, "/camera/camera/rgbd"
    )
    gesture_subscriber = message_filters.Subscriber(
        gesture_objectron_integrator, PoseArray, "/mediapipe/gesture/pointing_vector"
    )
    objectron_subscriber = message_filters.Subscriber(
        gesture_objectron_integrator, PolygonStamped, "/mediapipe/objectron/objects"
    )

    ts = message_filters.ApproximateTimeSynchronizer(
        [realsense_subscriber, gesture_subscriber, objectron_subscriber],
        10,
        0.1,
        allow_headerless=True,
    )
    ts.registerCallback(gesture_objectron_integrator.callback)

    rclpy.spin(gesture_objectron_integrator)

    rclpy.shutdown()


class GestureObjectronIntegrator(Node):
    def __init__(self):
        super().__init__("gesture_objectron_integrator")

    def callback(self, rgbd_msg, gesture_msg, objectron_msg):
        print("RGBD message: ", rgbd_msg)
        print("Gesture message: ", gesture_msg)
        print("Objectron message: ", objectron_msg)
