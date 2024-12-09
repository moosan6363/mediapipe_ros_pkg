import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from sensor_msgs.msg import Image
from sklearn.decomposition import PCA
from std_msgs.msg import Header
from tf2_geometry_msgs import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from mediapipe_ros_pkg.kalman_filter import KalmanFilter
from mediapipe_ros_pkg.realsense_subscriber import (
    RealsenseSubsctiber,
    estimate_object_points,
)
from mediapipe_ros_pkg.util import direction_vector_to_quaternion, write_poining_vector


def main(args=None):
    rclpy.init(args=args)
    camera_name = "left_camera"
    mediapipe_gesture_publisher = MediapipeHandPosePublisher(
        node_name="mediapipe_hand_pose_publisher",
        realsense_topic_name=f"/camera/{camera_name}/rgbd",
        annotated_image_topic_name="/mediapipe/hand/annotated_image",
        pointing_vector_topic_name="/mediapipe/hand/pose",
        source_frame_rel=f"{camera_name}_color_optical_frame",
        target_frame_rel="world",
        model_path=Path(
            "/home/ws/src/mediapipe_ros_pkg/models/gesture_recognizer.task"
        ),
        default_dtype=np.float32,
    )
    try:
        rclpy.spin(mediapipe_gesture_publisher)
    except KeyboardInterrupt:
        mediapipe_gesture_publisher.destroy_node()


class MediapipeHandPosePublisher(RealsenseSubsctiber):
    def __init__(
        self,
        node_name,
        realsense_topic_name,
        annotated_image_topic_name,
        pointing_vector_topic_name,
        source_frame_rel,
        target_frame_rel,
        model_path,
        default_dtype,
    ):
        super().__init__(node_name, realsense_topic_name)
        self.annotated_image_publisher = self.create_publisher(
            Image, annotated_image_topic_name, 10
        )
        self.pointing_vector_publisher = self.create_publisher(
            PoseStamped, pointing_vector_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame_rel = source_frame_rel
        self.target_frame_rel = target_frame_rel

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_path),
            num_hands=1,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.default_dtype = default_dtype

        self.bridge = CvBridge()

        self.enable_landmaks = [
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
        ]

        self.kalman_filter = KalmanFilter(
            dim_x=6,
            dim_z=3,
            h=np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ],
                dtype=self.default_dtype,
            ),
            x_0=np.zeros(6),
            p=np.eye(6) * 1.0,
            r=np.eye(3) * 0.05,
            q_var=1.0,
            f_func=lambda dt: np.array(
                [
                    [1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=self.default_dtype,
            ),
            verbose=True,
        )

        self.kf_dict = {
            landmark_idx: deepcopy(self.kalman_filter)
            for landmark_idx in self.enable_landmaks
        }

        self.pca = PCA(n_components=1)

    def callback(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8"),
        )

        # TODO : https://stackoverflow.com/questions/78841248/userwarning-symboldatabase-getprototype-is-deprecated-please-use-message-fac
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recognition_result = self.recognizer.recognize(mp_image)

        annotated_image = mp_image.numpy_view().copy()

        # TODO: only one hand
        if len(recognition_result.hand_landmarks) == 1:
            hand_landmarks = recognition_result.hand_landmarks[0]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            pose = self.pointing_vector_estimation(
                annotated_image,
                self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                rgbd_msg.rgb_camera_info,
                rgbd_msg.depth_camera_info,
                float(rgbd_msg.header.stamp.sec + rgbd_msg.header.stamp.nanosec * 1e-9),
                hand_landmarks_proto,
            )
            if pose is not None:
                pose_stamped = PoseStamped(
                    header=Header(
                        stamp=rgbd_msg.header.stamp,
                        frame_id=self.target_frame_rel,
                    ),
                    pose=pose,
                )
                self.pointing_vector_publisher.publish(pose_stamped)

        self.annotated_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

        return

    def pointing_vector_estimation(
        self,
        rgb_image,
        depth_image,
        rgb_camera_info,
        depth_camera_info,
        msg_timestamp,
        hand_landmarks_proto,
    ):
        image_points_dict = {}
        for landmark_idx in self.enable_landmaks:
            if (
                0 <= hand_landmarks_proto.landmark[landmark_idx].x <= 1
                and 0 <= hand_landmarks_proto.landmark[landmark_idx].y <= 1
            ):
                image_points_dict[landmark_idx] = (
                    int(
                        hand_landmarks_proto.landmark[landmark_idx].x
                        * (rgb_image.shape[1] - 1)
                    ),
                    int(
                        hand_landmarks_proto.landmark[landmark_idx].y
                        * (rgb_image.shape[0] - 1)
                    ),
                )
            else:
                image_points_dict[landmark_idx] = None

        previous_object_points_dict = {
            key: kf.z_hat for key, kf in self.kf_dict.items()
        }

        # Estimate 3D coordinates
        object_points_dict = estimate_object_points(
            image_points_dict,
            previous_object_points_dict,
            depth_image,
            rgb_camera_info,
        )

        # without Kalman filter
        pose = self.caluculate_pose(object_points_dict)
        if pose is not None:
            direction_vector, position = pose
            self.visualize_line(
                rgb_image,
                rgb_camera_info,
                image_points_dict,
                object_points_dict,
                direction_vector,
                position,
                (0, 0, 0),
            )

        # Apply Kalman filter
        for key, kf in self.kf_dict.items():
            kf.update(msg_timestamp, object_points_dict[key])
            object_points_dict[key] = kf.z_hat

        pose = self.caluculate_pose(object_points_dict)
        if pose is not None:
            direction_vector, position = pose
            self.visualize_line(
                rgb_image,
                rgb_camera_info,
                image_points_dict,
                object_points_dict,
                direction_vector,
                position,
                (0, 0, 255),
            )
            orientation = direction_vector_to_quaternion(direction_vector)
            position = Point(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            )
            pose = Pose(
                position=position,
                orientation=orientation,
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
                return
            pose = tf2_geometry_msgs.do_transform_pose(pose, transform)

            return pose
        else:
            return None

    def caluculate_pose(
        self,
        object_points_dict,
    ):
        object_points = np.array(
            [
                object_points
                for object_points in object_points_dict.values()
                if object_points is not None
            ],
            dtype=self.default_dtype,
        )
        if len(object_points) >= 4:
            self.pca.fit(object_points)
            direction_vector = self.pca.components_[0]
            position = self.pca.mean_

            position_to_tip = (
                object_points_dict[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                - position
            )

            if (
                np.linalg.norm(position_to_tip) != 0
                and np.linalg.norm(direction_vector) != 0
            ):
                cos_theta = np.dot(position_to_tip, direction_vector) / (
                    np.linalg.norm(position_to_tip) * np.linalg.norm(direction_vector)
                )
                if cos_theta < 0:
                    direction_vector = -direction_vector

                return direction_vector / np.linalg.norm(direction_vector), position
        else:
            return None

    def visualize_line(
        self,
        rgb_image,
        rgb_camera_info,
        image_points_dict,
        object_points_dict,
        direction_vector,
        position,
        color,
    ):
        image_points = np.array(
            [
                image_points
                for image_points in image_points_dict.values()
                if image_points is not None
            ],
            dtype=np.uint32,
        )
        object_points = np.array(
            [
                object_points
                for object_points in object_points_dict.values()
                if object_points is not None
            ],
            dtype=self.default_dtype,
        )

        k = np.array(rgb_camera_info.k).reshape(3, 3)
        d = np.array(rgb_camera_info.d)

        # Solve PnP
        try:
            retval, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points.astype(self.default_dtype),
                k,
                d,
                flags=cv2.SOLVEPNP_P3P,
            )
        except cv2.error:
            retval = False
            print("cv2.solvePnP failed")

        if retval:
            start_object_point = position
            end_object_point = position + direction_vector * 0.4

            start_image_point, _ = cv2.projectPoints(
                start_object_point,
                rvec,
                tvec,
                k,
                d,
            )

            end_image_point, _ = cv2.projectPoints(end_object_point, rvec, tvec, k, d)

            write_poining_vector(
                rgb_image, start_image_point[0][0], end_image_point[0][0], color
            )
