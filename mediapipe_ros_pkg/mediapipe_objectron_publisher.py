import mediapipe as mp
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    Quaternion,
    Vector3,
)
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from mediapipe_ros_pkg.realsense_subscriber import (
    RealsenseSubsctiber,
    estimate_object_points,
)


def main(args=None):
    rclpy.init(args=args)
    camera_name = "front_camera"
    mediapipe_objectron_publisher = MediaPipeObjectronPublisher(
        node_name="mediapipe_objectron_publisher",
        realsense_topic_name=f"/camera/{camera_name}/rgbd",
        annotated_image_topic_name="/mediapipe/objectron/annotated_image",
        objectron_marker_array_topic_name="/mediapipe/objectron/marker_array",
        source_frame_rel=f"{camera_name}_color_optical_frame",
        target_frame_rel=f"{camera_name}_color_frame",
    )
    try:
        rclpy.spin(mediapipe_objectron_publisher)
    except KeyboardInterrupt:
        mediapipe_objectron_publisher.destroy_node()


class MediaPipeObjectronPublisher(RealsenseSubsctiber):
    def __init__(
        self,
        node_name,
        realsense_topic_name,
        annotated_image_topic_name,
        objectron_marker_array_topic_name,
        source_frame_rel,
        target_frame_rel,
    ):
        super().__init__(node_name, realsense_topic_name)
        self.annotated_image_publisher = self.create_publisher(
            Image, annotated_image_topic_name, 10
        )
        self.objectron_marker_array_publisher = self.create_publisher(
            MarkerArray, objectron_marker_array_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame_rel = source_frame_rel
        self.target_frame_rel = target_frame_rel

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
                markers = []
                for idx, detected_object in enumerate(results.detected_objects):
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        detected_object.landmarks_2d,
                        self.mp_objectron.BOX_CONNECTIONS,
                    )

                    marker = self.create_marker(
                        rgbd_msg, detected_object.landmarks_2d.landmark, idx
                    )
                    if marker is not None:
                        markers.append(marker)

                if len(markers) > 0:
                    self.objectron_marker_array_publisher.publish(
                        MarkerArray(markers=markers)
                    )

            self.annotated_image_publisher.publish(
                self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
            )
            return

    def create_marker(self, rgbd_msg, landmark_2d, id):
        image_points_dict = {}
        image_points_dict[self.CUP_CENTER_IDX] = (
            int(landmark_2d[self.CUP_CENTER_IDX].x * (rgbd_msg.rgb.width - 1)),
            int(landmark_2d[self.CUP_CENTER_IDX].y * (rgbd_msg.rgb.height - 1)),
        )

        # TODO
        previous_object_points_dict = {}
        previous_object_points_dict[self.CUP_CENTER_IDX] = (0, 0, 0.5)

        object_points_dict = estimate_object_points(
            image_points_dict,
            previous_object_points_dict,
            self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
            rgbd_msg.rgb_camera_info,
        )

        diameter = 0.1
        height = 0.06

        if object_points_dict[self.CUP_CENTER_IDX] is None:
            return None
        else:
            point_stamped = PointStamped(
                header=Header(frame_id=self.source_frame_rel),
                point=Point(
                    x=object_points_dict[self.CUP_CENTER_IDX][0],
                    y=object_points_dict[self.CUP_CENTER_IDX][1],
                    z=object_points_dict[self.CUP_CENTER_IDX][2],
                ),
            )
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame_rel,
                    self.source_frame_rel,
                    rclpy.time.Time(),
                    timeout=rclpy.time.Duration(seconds=1),
                )
            except Exception as e:
                print(e)
                return None
            point_stamped = tf2_geometry_msgs.do_transform_point(
                point_stamped, transform
            )

            marker = Marker(
                header=Header(
                    stamp=rgbd_msg.rgb.header.stamp,
                    frame_id=self.target_frame_rel,
                ),
                id=id,
                type=Marker.CYLINDER,
                action=Marker.ADD,
                pose=Pose(
                    position=Point(
                        x=point_stamped.point.x + diameter / 2,
                        y=point_stamped.point.y,
                        z=point_stamped.point.z,
                    ),
                    orientation=Quaternion(),
                ),
                scale=Vector3(
                    x=diameter,
                    y=diameter,
                    z=height,
                ),
                color=ColorRGBA(
                    r=1.0,
                    g=0.0,
                    b=0.0,
                    a=0.5,
                ),
            )
            return marker
