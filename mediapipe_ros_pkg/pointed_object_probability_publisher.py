import message_filters
import numpy as np
import rclpy
import scipy.stats
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float32MultiArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray

from mediapipe_ros_pkg.util import (
    quarterion_to_direction_vector,
)


def main(args=None):
    rclpy.init(args=args)
    pointed_object_probability_publisher = PointedObjectProbabilityPublisher(
        node_name="pointed_object_probability_publisher",
        hand_pose_topic_name="/mediapipe/hand/pose",
        head_pose_topic_name="/sixdrepnet/head/pose",
        objectron_topic_name="/mediapipe/objectron/marker_array",
        average_pose_topic_name="/mediapipe/pointed_object/average_pose",
        probability_topic_name="/mediapipe/pointed_object/probability",
        hand_distance_topic_name="/mediapipe/pointed_object/hand_distance",
        head_distance_topic_name="/mediapipe/pointed_object/head_distance",
        marker_array_topic_name="/mediapipe/pointed_object/marker_array",
        target_frame_rel="front_camera_color_frame",
        gaussian_sigma=1.0,
    )
    try:
        rclpy.spin(pointed_object_probability_publisher)
    except KeyboardInterrupt:
        pointed_object_probability_publisher.destroy_node()


class PointedObjectProbabilityPublisher(Node):
    def __init__(
        self,
        node_name,
        hand_pose_topic_name,
        head_pose_topic_name,
        objectron_topic_name,
        average_pose_topic_name,
        probability_topic_name,
        hand_distance_topic_name,
        head_distance_topic_name,
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

        self.average_pose_pulisher = self.create_publisher(
            PoseStamped, average_pose_topic_name, 10
        )
        self.probability_publisher = self.create_publisher(
            Float32MultiArray, probability_topic_name, 10
        )
        self.hand_distance_publisher = self.create_publisher(
            Float32MultiArray, hand_distance_topic_name, 10
        )
        self.head_distance_publisher = self.create_publisher(
            Float32MultiArray, head_distance_topic_name, 10
        )
        self.marker_array_publisher = self.create_publisher(
            MarkerArray, marker_array_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame_rel = target_frame_rel

        self.gaussian_sigma = gaussian_sigma

        self.objectron_msg = None

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

        hand_eds = []
        head_eds = []
        ucps = []
        markers = []
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

            # print(ed_hand, ed_head)

            hand_eds.append(ed_hand)
            head_eds.append(ed_head)
            ucps.append(ucp)
            marker.color = ColorRGBA(
                r=1.0 * ucp,
                b=1.0 * (1 - ucp),
                a=0.5,
            )
            markers.append(marker)
        print(ucps)

        hand_ed_msg = Float32MultiArray(data=hand_eds)
        head_ed_msg = Float32MultiArray(data=head_eds)
        pd_msg = Float32MultiArray(data=ucps)
        marker_array = MarkerArray(markers=markers)
        self.probability_publisher.publish(pd_msg)
        self.hand_distance_publisher.publish(hand_ed_msg)
        self.head_distance_publisher.publish(head_ed_msg)
        self.marker_array_publisher.publish(marker_array)

    def objectron_callback(self, objectron_msg):
        self.objectron_msg = objectron_msg

    def get_average_line(self, p1, d1, p2, d2):
        d1 = self.normalize(d1)
        d2 = self.normalize(d2)
        d = self.normalize(d1 + d2)

        # caluculate the nearest point
        n = np.cross(d1, d2)
        n = self.normalize(n)
        t = np.dot(np.cross(p2 - p1, d2), n) / np.dot(n, n)
        s = np.dot(np.cross(p2 - p1, d1), n) / np.dot(n, n)

        p = (p1 + t * d1 + p2 + s * d2) / 2

        return p, d

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
