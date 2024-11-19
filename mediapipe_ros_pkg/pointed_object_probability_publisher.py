import message_filters
import numpy as np
import rclpy
import scipy.stats
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import MarkerArray

from mediapipe_ros_pkg.util import quarterion_to_direction_vector


def main(args=None):
    rclpy.init(args=args)
    pointed_object_probability_publisher = PointedObjectProbabilityPublisher(
        node_name="pointed_object_probability_publisher",
        gesture_topic_name="/mediapipe/gesture/pointing_vector",
        objectron_topic_name="/mediapipe/objectron/marker_array",
        probability_topic_name="/mediapipe/pointed_object/probability",
        distance_topic_name="/mediapipe/pointed_object/distance",
        marker_array_topic_name="/mediapipe/pointed_object/marker_array",
    )
    try:
        rclpy.spin(pointed_object_probability_publisher)
    except KeyboardInterrupt:
        pointed_object_probability_publisher.destroy_node()


class PointedObjectProbabilityPublisher(Node):
    def __init__(
        self,
        node_name,
        gesture_topic_name,
        objectron_topic_name,
        probability_topic_name,
        distance_topic_name,
        marker_array_topic_name,
    ):
        super().__init__(node_name)

        gesture_subscriber = message_filters.Subscriber(
            self, PoseArray, gesture_topic_name
        )
        objectron_subscriber = message_filters.Subscriber(
            self, MarkerArray, objectron_topic_name
        )
        ts = message_filters.ApproximateTimeSynchronizer(
            [gesture_subscriber, objectron_subscriber], 10, 1.0, allow_headerless=True
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

    def callback(self, gesture_msg, objectron_msg):
        # TODO: Index 0 is used. But, it should be changed.
        gesture_position = gesture_msg.poses[0].position
        gesture_orientation = gesture_msg.poses[0].orientation
        gesture_position = np.array(
            [gesture_position.x, gesture_position.y, gesture_position.z]
        )
        gesture_orientation = np.array(
            [
                gesture_orientation.x,
                gesture_orientation.y,
                gesture_orientation.z,
                gesture_orientation.w,
            ]
        )
        gesture_orientation = quarterion_to_direction_vector(gesture_orientation)

        eds = []
        ucps = []
        markers = []
        for marker in objectron_msg.markers:
            object_point = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
            )
            AP = object_point - gesture_position
            euclidean_distance = np.linalg.norm(
                np.cross(AP, gesture_orientation)
            ) / np.linalg.norm(gesture_orientation)

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
