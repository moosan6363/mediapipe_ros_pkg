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

from mediapipe_ros_pkg.util import quarterion_to_direction_vector


def main(args=None):
    rclpy.init(args=args)
    pointed_object_probability_publisher = PointedObjectProbabilityPublisher(
        node_name="pointed_object_probability_publisher",
        hand_pose_topic_name="/mediapipe/gesture/pointing_vector",
        head_pose_topic_name="/mediapipe/head/head_direction_vector",
        objectron_topic_name="/mediapipe/objectron/marker_array",
        probability_topic_name="/mediapipe/pointed_object/probability",
        distance_topic_name="/mediapipe/pointed_object/distance",
        marker_array_topic_name="/mediapipe/pointed_object/marker_array",
        target_frame_rel="front_camera_color_frame",
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
        probability_topic_name,
        distance_topic_name,
        marker_array_topic_name,
        target_frame_rel,
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
            queue_size=20,
            slop=1.0,
            allow_headerless=False,
        )
        ts.registerCallback(self.callback)

        self.probability_publisher = self.create_publisher(
            Float32MultiArray, probability_topic_name, 10
        )
        self.distance_publisher = self.create_publisher(
            Float32MultiArray, distance_topic_name, 10
        )
        self.marker_array_publisher = self.create_publisher(
            MarkerArray, marker_array_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame_rel = target_frame_rel

        self.objectron_msg = None

    def callback(self, hand_pose_msg, head_pose_msg):
        print("callback")
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
        hand_pose_position = hand_pose.position
        hand_pose_orientation = hand_pose.orientation
        hand_pose_position = np.array(
            [hand_pose_position.x, hand_pose_position.y, hand_pose_position.z]
        )
        hand_pose_orientation = np.array(
            [
                hand_pose_orientation.x,
                hand_pose_orientation.y,
                hand_pose_orientation.z,
                hand_pose_orientation.w,
            ]
        )
        hand_pose_orientation = quarterion_to_direction_vector(hand_pose_orientation)

        eds = []
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
            AP = object_point - hand_pose_position
            euclidean_distance = np.linalg.norm(
                np.cross(AP, hand_pose_orientation)
            ) / np.linalg.norm(hand_pose_orientation)

            # Because of the assumption that the object is a cylinder, the distance is calculated from the center of the object
            euclidean_distance = max(0, euclidean_distance - marker.scale.x / 2)

            upper_cumulative_probability = (
                1 - scipy.stats.norm.cdf(euclidean_distance, loc=0, scale=0.1)
            ) * 2

            eds.append(euclidean_distance)
            ucps.append(upper_cumulative_probability)
            marker.color = ColorRGBA(
                r=1.0 * upper_cumulative_probability,
                b=1.0 * (1 - upper_cumulative_probability),
                a=0.5,
            )
            markers.append(marker)

        ed_msg = Float32MultiArray(data=eds)
        pd_msg = Float32MultiArray(data=ucps)
        marker_array = MarkerArray(markers=markers)
        self.probability_publisher.publish(pd_msg)
        self.distance_publisher.publish(ed_msg)
        self.marker_array_publisher.publish(marker_array)

    def objectron_callback(self, objectron_msg):
        self.objectron_msg = objectron_msg
