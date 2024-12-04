from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from mediapipe.framework.formats import landmark_pb2
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from mediapipe_ros_pkg.realsense_subscriber import (
    RealsenseSubsctiber,
    estimate_object_points,
)
from mediapipe_ros_pkg.util import (
    rmat_to_quaternion,
    write_poining_vector,
)


def main(args=None):
    rclpy.init(args=args)
    camera_name = "side_camera"
    mediapipe_gesture_publisher = MediapipeHeadPosePublisher(
        node_name="mediapipe_head_pose_publisher",
        realsense_topic_name=f"/camera/{camera_name}/rgbd",
        annotated_image_topic_name="/mediapipe/head/annotated_image",
        head_pose_topic_name="/mediapipe/head/pose",
        source_frame_rel=f"{camera_name}_color_optical_frame",
        target_frame_rel=f"{camera_name}_color_frame",
        model_path=Path(
            "/home/ws/src/mediapipe_ros_pkg/models/face_landmarker_v2_with_blendshapes.task"
        ),
        default_dtype=np.float32,
    )
    try:
        rclpy.spin(mediapipe_gesture_publisher)
    except KeyboardInterrupt:
        mediapipe_gesture_publisher.destroy_node()


class MediapipeHeadPosePublisher(RealsenseSubsctiber):
    def __init__(
        self,
        node_name,
        realsense_topic_name,
        annotated_image_topic_name,
        head_pose_topic_name,
        source_frame_rel,
        target_frame_rel,
        model_path,
        default_dtype,
    ):
        super().__init__(node_name, realsense_topic_name)
        self.annotated_image_publisher = self.create_publisher(
            Image, annotated_image_topic_name, 10
        )
        self.head_pose_publisher = self.create_publisher(
            PoseStamped, head_pose_topic_name, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame_rel = source_frame_rel
        self.target_frame_rel = target_frame_rel

        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            num_faces=1,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(self.options)

        self.default_dtype = default_dtype

        self.bridge = CvBridge()

        # self.object_points_dict = {
        #     1: [0.285, 0.528, 0.200],
        #     9: [0.285, 0.371, 0.152],
        #     57: [0.197, 0.574, 0.128],
        #     130: [0.173, 0.425, 0.108],
        #     287: [0.360, 0.574, 0.128],
        #     359: [0.391, 0.425, 0.108],
        # }

        self.landmark_indices = {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_left_corner": 226,
            "right_eye_right_corner": 446,
            "left_Mouth_corner": 57,
            "right_mouth_corner": 287,
            "between_eyes": 6,
        }

        self.object_points_dict = {
            "nose_tip": [0.0, 0.0, 0.0],
            "chin": [0.0, -0.330, -0.065],
            "left_eye_left_corner": [-0.225, 0.17, -0.135],
            "right_eye_right_corner": [0.225, 0.17, -0.135],
            "left_Mouth_corner": [-0.15, -0.15, -0.125],
            "right_mouth_corner": [0.15, -0.15, -0.125],
            "between_eyes": [0.0, 0.17, -0.135],
        }

    def callback(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")
        annotated_image = image.copy()

        face_landmarks_proto = self.detect(image)
        # TODO: only one face is detected
        if face_landmarks_proto is not None:
            annotated_image = self.draw_landmarks(annotated_image, face_landmarks_proto)

            pose = self.head_direction_vector_estimation(
                rgb_image=annotated_image,
                depth_image=rgbd_msg.depth,
                rgb_camera_info=rgbd_msg.rgb_camera_info,
                depth_camera_info=rgbd_msg.depth_camera_info,
                face_landmarks_proto=face_landmarks_proto,
            )

            if pose is not None:
                pose_stamped = PoseStamped(
                    header=Header(
                        stamp=rgbd_msg.header.stamp,
                        frame_id=self.target_frame_rel,
                    ),
                    pose=pose,
                )
                self.head_pose_publisher.publish(pose_stamped)

        self.annotated_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

        return

    def detect(self, image):
        height, width, _ = image.shape
        mid_height, mid_width = height // 2, width // 2
        quarter_height, quarter_width = height // 4, width // 4

        crop_images = np.array(
            [
                image[:mid_height, :mid_width],  # top-left
                image[:mid_height, quarter_width : quarter_width * 3],  # top-center
                image[:mid_height, mid_width:],  # top-right
                image[quarter_height : quarter_height * 3, :mid_width],  # center-left
                image[
                    quarter_height : quarter_height * 3,
                    quarter_width : quarter_width * 3,
                ],  # center-center
                image[quarter_height : quarter_height * 3, mid_width:],  # center-right
                image[mid_height:, :mid_width],  # bottom-left
                image[mid_height:, quarter_width : quarter_width * 3],  # bottom-center
                image[mid_height:, mid_width:],  # bottom-right
            ]
        )

        def x_coordinate(i, x):
            if i % 3 == 0:
                return x / 2
            elif i % 3 == 1:
                return (2 * x + 1) / 4
            elif i % 3 == 2:
                return (x + 1) / 2

        def y_coordinate(i, y):
            if i < 3:
                return y / 2
            elif i < 6:
                return (2 * y + 1) / 4
            elif i < 9:
                return (y + 1) / 2

        for i, crop_image in enumerate(crop_images):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_image)
            detection_result = self.detector.detect(mp_image)
            if len(detection_result.face_landmarks) > 0:
                face_landmarks = detection_result.face_landmarks[0]
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=x_coordinate(i, landmark.x),
                            y=y_coordinate(i, landmark.y),
                            z=landmark.z,
                        )
                        for landmark in face_landmarks
                    ]
                )
                return face_landmarks_proto
        return None

    def head_direction_vector_estimation(
        self,
        rgb_image,
        depth_image,
        rgb_camera_info,
        depth_camera_info,
        face_landmarks_proto,
    ):
        image_points_dict = {}
        for key in self.object_points_dict.keys():
            idx = self.landmark_indices[key]
            image_point = (
                int(face_landmarks_proto.landmark[idx].x * rgb_camera_info.width - 1),
                int(face_landmarks_proto.landmark[idx].y * rgb_camera_info.height - 1),
            )
            image_points_dict[key] = image_point

        image_points = np.array(
            list(image_points_dict.values()), dtype=self.default_dtype
        )

        object_points = np.array(
            list(self.object_points_dict.values()), dtype=self.default_dtype
        )

        k = np.array(rgb_camera_info.k).reshape(3, 3)
        d = np.array(rgb_camera_info.d)

        retval, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points, image_points, k, d, flags=cv2.SOLVEPNP_P3P
        )

        if retval:
            self.visualize_line(rgb_image, rvec, tvec, k, d, (0, 255, 0))
            rmat, _ = cv2.Rodrigues(rvec)

            orientation = rmat_to_quaternion(rmat)
            orientation = Quaternion(
                x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
            )

            ###--------------------------------------
            # Option1 for Position Estimation
            # position = (
            #     np.dot(rmat, self.object_points_dict["between_eyes"]) + tvec.flatten()
            # )

            # position = Point(
            #     x=float(position[0]),
            #     y=float(position[1]),
            #     z=float(position[2]),
            # )

            # Option2 for Position Estimation
            previous_object_points_dict = {}
            previous_object_points_dict["between_eyes"] = (0, 0, 0.5)

            object_points_dict = estimate_object_points(
                {"between_eyes": image_points_dict["between_eyes"]},
                previous_object_points_dict,
                self.bridge.imgmsg_to_cv2(depth_image, "passthrough"),
                depth_camera_info,
            )
            position = Point(
                x=object_points_dict["between_eyes"][0],
                y=object_points_dict["between_eyes"][1],
                z=object_points_dict["between_eyes"][2],
            )

            ###--------------------------------------

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
            from tf2_geometry_msgs import tf2_geometry_msgs

            pose = tf2_geometry_msgs.do_transform_pose(pose, transform)

            ###--------------------------------------
            # TODO: To correct because the orientation does not match the ROS2 coordinate system
            from scipy.spatial.transform import Rotation

            from mediapipe_ros_pkg.util import quaternion_multiply

            rotation = Rotation.from_matrix(
                np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            ).as_quat()

            rotation = Quaternion(
                x=rotation[0],
                y=rotation[1],
                z=rotation[2],
                w=rotation[3],
            )

            orientation = quaternion_multiply(orientation, rotation)
            pose = Pose(
                position=pose.position,
                orientation=orientation,
            )
            ###--------------------------------------

            return pose
        else:
            return None

    def draw_landmarks(self, annotated_image, face_landmarks_proto):
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        return annotated_image

    def visualize_line(self, rgb_image, rvec, tvec, k, d, color):
        start_object_point = np.array(self.object_points_dict["between_eyes"])
        end_object_point = start_object_point + np.array(
            [0, 0, 1], dtype=self.default_dtype
        )

        start_image_point, _ = cv2.projectPoints(start_object_point, rvec, tvec, k, d)

        end_image_point, _ = cv2.projectPoints(end_object_point, rvec, tvec, k, d)

        write_poining_vector(
            rgb_image,
            start_image_point[0][0],
            end_image_point[0][0],
            color,
        )