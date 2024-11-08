import mediapipe as mp
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, Polygon, PolygonStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from mediapipe_ros_pkg.realsense import estimate_object_points
from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber


def main(args=None):
    rclpy.init(args=args)

    mediapipe_objectron_publisher = MediaPipeObjectronPublisher()
    realsense_subscriber = RealsenseSubsctiber(mediapipe_objectron_publisher.callback)

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    mediapipe_objectron_publisher.destroy_node()
    rclpy.shutdown()


class MediaPipeObjectronPublisher(Node):
    def __init__(self):
        super().__init__("mediapipe_objectron_publisher")
        self.objectron_image_publisher = self.create_publisher(
            Image, "/mediapipe/objectron/annotated_image", 10
        )
        self.objectron_objects_publisher = self.create_publisher(
            PolygonStamped, "/mediapipe/objectron/objects", 10
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_objectron = mp.solutions.objectron
        self.bridge = CvBridge()
        self.CUP_CENTER_IDX = 0

    def callback(self, rgbd_msg):
        with self.mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.99,
            model_name="Cup",
            focal_length=(rgbd_msg.rgb_camera_info.k[0], rgbd_msg.rgb_camera_info.k[4]),
            principal_point=(
                rgbd_msg.rgb_camera_info.k[2],
                rgbd_msg.rgb_camera_info.k[5],
            ),
            image_size=(rgbd_msg.rgb.width, rgbd_msg.rgb.height),
        ) as objectron:
            rgb_image_msg = rgbd_msg.rgb
            image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            results = objectron.process(image)

            # Draw box landmarks.
            annotated_image = image.copy()
            if results.detected_objects:
                points = []
                for detected_object in results.detected_objects:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        detected_object.landmarks_2d,
                        self.mp_objectron.BOX_CONNECTIONS,
                    )
                    self.mp_drawing.draw_axis(
                        annotated_image,
                        detected_object.rotation,
                        detected_object.translation,
                    )

                    image_point = (
                        int(
                            detected_object.landmarks_2d.landmark[self.CUP_CENTER_IDX].x
                            * (rgbd_msg.rgb.width - 1)
                        ),
                        int(
                            detected_object.landmarks_2d.landmark[self.CUP_CENTER_IDX].y
                            * (rgbd_msg.rgb.height - 1)
                        ),
                    )

                    image_points_dict = {}
                    image_points_dict[self.CUP_CENTER_IDX] = image_point

                    # TODO
                    previous_object_points_dict = {}
                    previous_object_points_dict[self.CUP_CENTER_IDX] = (0, 0, 0.5)

                    object_points = estimate_object_points(
                        image_points_dict,
                        previous_object_points_dict,
                        self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                        rgbd_msg.depth_camera_info,
                    )

                    points.append(
                        Point32(
                            x=object_points[self.CUP_CENTER_IDX][0],
                            y=object_points[self.CUP_CENTER_IDX][1],
                            z=object_points[self.CUP_CENTER_IDX][2],
                        )
                    )
                self.objectron_objects_publisher.publish(
                    PolygonStamped(
                        header=rgb_image_msg.header, polygon=Polygon(points=points)
                    )
                )

            self.objectron_image_publisher.publish(
                self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
            )


if __name__ == "__main__":
    mp_objectron = mp.solutions.objectron
    mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.99,
        model_name="Cup",
    )
