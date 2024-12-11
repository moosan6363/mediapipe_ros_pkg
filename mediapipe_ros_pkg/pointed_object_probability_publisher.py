import message_filters
import numpy as np
import rclpy
import scipy.stats
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Empty
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray
import os

from mediapipe_ros_pkg.util import (
    quarterion_to_direction_vector,
)

from mediapipe_ros_pkg.reduced_logic import ControlLogic
from rclpy.executors import MultiThreadedExecutor

import traceback as tb


def main(args=None):
    rclpy.init(args=args)
    pointed_object_probability_publisher = PointedObjectProbabilityPublisher(
        node_name="pointed_object_probability_publisher",
        hand_pose_topic_name="/mediapipe/hand/pose",
        head_pose_topic_name="/sixdrepnet/head/pose",
        objectron_topic_name="/yolo/object_detection/marker_array",
        send_action_topic_name="/pointed_object/send_action",
        marker_array_topic_name="/pointed_object/marker_array",
        target_frame_rel="world",
        gaussian_sigma=1.0,
    )
    pointed_object_probability_publisher.get_logger().info("-----------------")
    pointed_object_probability_publisher.get_logger().info(os.environ["ROS_DOMAIN_ID"])
    cl = ControlLogic(node=pointed_object_probability_publisher)
    pointed_object_probability_publisher.register_pass_callback(cl.sendPassAction)
    cl.node.get_logger().info("Init done.")
    try:
        mte = MultiThreadedExecutor(4)
        # cl.sendPassAction([0.5, 0.0, 0.02], [0.1, 0.1, 0.1])
        rclpy.spin(pointed_object_probability_publisher, executor=mte)
        # rclpy.spin(cl, executor=mte)
        # rclpy.spin_once(cl, executor=mte)
        cl.node.get_logger().info("ready")
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()

    pointed_object_probability_publisher.destroy_node()
    # cl.destroy_node()


class PointedObjectProbabilityPublisher(Node):
    def __init__(
        self,
        node_name,
        hand_pose_topic_name,
        head_pose_topic_name,
        objectron_topic_name,
        send_action_topic_name,
        marker_array_topic_name,
        target_frame_rel,
        gaussian_sigma,
    ):
        super().__init__(node_name)

        # self.hand_pose_subscription = self.create_subscription(
        #     PoseStamped, hand_pose_topic_name, self.hand_pose_callback, 10
        # )
        # self.head_pose_subscription = self.create_subscription(
        #     PoseStamped, head_pose_topic_name, self.head_pose_callback, 10
        # )
        self.objectron_subscription = self.create_subscription(
            MarkerArray, objectron_topic_name, self.objectron_callback, 10
        )

        self.send_action_subscription = self.create_subscription(
            Empty, send_action_topic_name, self.send_action_callback, 10
        )

        hand_pose_subscriber = message_filters.Subscriber(
            self, PoseStamped, hand_pose_topic_name
        )
        head_pose_subscriber = message_filters.Subscriber(
            self, PoseStamped, head_pose_topic_name
        )
        # objectron_subscriber = message_filters.Subscriber(
        #     self, MarkerArray, objectron_topic_name
        # )
        ts = message_filters.ApproximateTimeSynchronizer(
            fs=[
                hand_pose_subscriber,
                head_pose_subscriber,
            ],
            queue_size=50,
            slop=1.0,
            allow_headerless=False,
        )
        ts.registerCallback(self.callback)

        self.marker_array_publisher = self.create_publisher(
            MarkerArray, marker_array_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame_rel = target_frame_rel

        self.gaussian_sigma = gaussian_sigma

        self.objectron_msg = None

        self.object_points = []

    def register_pass_callback(self, send_pass_action_func):
        self.send_pass_action_func = send_pass_action_func

    def callback(self, hand_pose_msg, head_pose_msg):
        if self.objectron_msg is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_rel,
                hand_pose_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=1),
            )
            hand_pose = tf2_geometry_msgs.do_transform_pose(
                hand_pose_msg.pose, transform
            )

            transform = self.tf_buffer.lookup_transform(
                self.target_frame_rel,
                head_pose_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=1),
            )
            head_pose = tf2_geometry_msgs.do_transform_pose(
                head_pose_msg.pose, transform
            )
        except Exception as e:
            print(e)
            return

        hand_pose_position = np.array(
            [hand_pose.position.x, hand_pose.position.y, hand_pose.position.z]
        )
        hand_pose_orientation = quarterion_to_direction_vector(hand_pose.orientation)

        head_pose_position = np.array(
            [head_pose.position.x, head_pose.position.y, head_pose.position.z]
        )
        head_pose_orientation = quarterion_to_direction_vector(head_pose.orientation)

        markers = []

        self.object_points = []

        for marker in self.objectron_msg.markers:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame_rel,
                    marker.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.time.Duration(seconds=1),
                )
                objectron_pose = tf2_geometry_msgs.do_transform_pose(
                    marker.pose, transform
                )
            except Exception as e:
                print(e)
                return

            object_point = np.array(
                [
                    objectron_pose.position.x,
                    objectron_pose.position.y,
                    objectron_pose.position.z,
                ]
            )

            ed_hand, ucp_hand = self.get_ed_and_ucp(
                hand_pose_position, hand_pose_orientation, object_point, marker
            )

            ed_head, ucp_head = self.get_ed_and_ucp(
                head_pose_position, head_pose_orientation, object_point, marker
            )

            ucp = ucp_hand * ucp_head

            self.object_points.append(
                {
                    "id": marker.id,
                    "object_point": object_point,
                    "ucp": ucp,
                }
            )

            # self.get_logger().info(
            #     f"id: {marker.id}, x: {object_point[0]:.2f}, y: {object_point[1]:.2f}, z: {object_point[2]:.2f}, ucp: {ucp:.2f}, ucp_hand: {ucp_hand:.2f}, ucp_head: {ucp_head:.2f}"
            # )

            marker.color = ColorRGBA(
                r=1.0 * ucp,
                b=1.0 * (1 - ucp),
                a=0.5,
            )
            markers.append(marker)

        self.marker_array_publisher.publish(MarkerArray(markers=markers))

    def send_action_callback(self, msg):
        if self.object_points is None:
            self.get_logger().error("No pointed object.")
        else:
            for object_point in self.object_points:
                self.get_logger().info(
                    f"id: {object_point['id']}, ucp: {object_point['ucp']:.2f}, x: {object_point['object_point'][0]:.2f}, y: {object_point['object_point'][1]:.2f}, z: {object_point['object_point'][2]:.2f}"
                )
            max_ucp_object = max(self.object_points, key=lambda x: x["ucp"])
            self.get_logger().info(
                f"send action: x: {max_ucp_object['object_point'][0]:.2f}, y: {max_ucp_object['object_point'][1]:.2f}, z: {max_ucp_object['object_point'][2]:.2f}"
            )
            self.send_pass_action_func(max_ucp_object["object_point"], [0.1, 0.1, 0.1])

    # def send_action_callback(self, msg):

    def objectron_callback(self, objectron_msg):
        self.objectron_msg = objectron_msg

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_ed_and_ucp(self, p, d, object_point, marker):
        AP = object_point - p
        euclidean_distance = np.linalg.norm(np.cross(AP, d)) / np.linalg.norm(d)

        # Because of the assumption that the object is a cylinder, the distance is calculated from the center of the object
        euclidean_distance = max(0, euclidean_distance - marker.scale.x / 2)

        upper_cumulative_probability = (
            1
            - scipy.stats.norm.cdf(euclidean_distance, loc=0, scale=self.gaussian_sigma)
        ) * 2

        return euclidean_distance, upper_cumulative_probability
