import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.decomposition import PCA

from mediapipe_ros_pkg.kalman_filter import KalmanFilter
from mediapipe_ros_pkg.realsense import estimate_object_points
from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber


def main(args=None):
    rclpy.init(args=args)

    mediapipe_gesture_publisher = MediapipeGesturePublisher()
    realsense_subscriber = RealsenseSubsctiber(mediapipe_gesture_publisher.forward)

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    mediapipe_gesture_publisher.destroy_node()
    rclpy.shutdown()


class MediapipeGesturePublisher(Node):
    def __init__(self):
        super().__init__("mediapipe_gesture_publisher")
        self.gesture_image_publisher = self.create_publisher(
            Image, "/mediapipe/gesture/annotated_image", 10
        )

        # TODO: Fix hard code
        mediapipe_model_path = Path(
            "/home/ws/src/mediapipe_ros_pkg/models/gesture_recognizer.task"
        )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.base_options = python.BaseOptions(mediapipe_model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options, num_hands=1
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.dtype = np.float32

        self.bridge = CvBridge()

        self.exposed_landmarks = [
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
                dtype=self.dtype,
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
                dtype=self.dtype,
            ),
            verbose=True,
        )

        self.kf_dict = {
            landmark_idx: deepcopy(self.kalman_filter)
            for landmark_idx in self.exposed_landmarks
        }

    def forward(self, rgbd_msg):
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

        for hand_landmarks in recognition_result.hand_landmarks:
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

            self.pointing_vector_estimation(
                annotated_image,
                self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                rgbd_msg.rgb_camera_info,
                rgbd_msg.depth_camera_info,
                float(rgbd_msg.header.stamp.sec + rgbd_msg.header.stamp.nanosec * 1e-9),
                hand_landmarks_proto,
            )

        self.gesture_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

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
        for landmark_idx in self.exposed_landmarks:
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
            depth_camera_info,
        )

        self.visualize_line(
            rgb_image,
            rgb_camera_info,
            image_points_dict,
            object_points_dict,
            (0, 0, 0),
        )

        # Apply Kalman filter
        for key, kf in self.kf_dict.items():
            kf.update(msg_timestamp, object_points_dict[key])
            object_points_dict[key] = kf.z_hat

        self.visualize_line(
            rgb_image,
            rgb_camera_info,
            image_points_dict,
            object_points_dict,
            (0, 0, 255),
        )

    def pca(self, data):
        pca = PCA(n_components=1)
        pca.fit(data)

        return pca.components_[0], pca.mean_

    def visualize_line(
        self,
        rgb_image,
        rgb_camera_info,
        image_points_dict,
        object_points_dict,
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
            dtype=self.dtype,
        )

        k = np.array(rgb_camera_info.k).reshape(3, 3)
        d = np.array(rgb_camera_info.d)

        # Solve PnP
        try:
            retval, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points.astype(self.dtype),
                k,
                d,
                flags=cv2.SOLVEPNP_P3P,
            )
        except cv2.error:
            retval = False
            print("cv2.solvePnP failed")

        if retval:
            direction_vector, mean = self.pca(object_points)

            # 0.3m away from the mean point
            start_object_point = mean + direction_vector * 0.3
            end_object_point = mean - direction_vector * 0.3

            # set start point self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            start_point, _ = cv2.projectPoints(
                start_object_point,
                rvec,
                tvec,
                k,
                d,
            )

            end_point, _ = cv2.projectPoints(end_object_point, rvec, tvec, k, d)

            self.write_poining_vector(
                rgb_image, start_point[0][0], end_point[0][0], color
            )

    def write_poining_vector(self, image, start_point, end_point, color):
        cv2.line(
            image,
            (int(start_point[0]), int(start_point[1])),
            (int(end_point[0]), int(end_point[1])),
            color,
            3,
        )
