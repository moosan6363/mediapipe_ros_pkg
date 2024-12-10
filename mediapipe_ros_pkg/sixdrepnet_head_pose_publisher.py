import time
from math import cos, sin
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.logging
import torch
from cv_bridge import CvBridge
from face_detection import RetinaFace
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from PIL import Image as PILImage
from pyrealsense2 import intrinsics, rs2_deproject_pixel_to_point
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from sixdrepnet.model import SixDRepNet
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from torch.backends import cudnn
from torchvision import transforms

from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber


def main(args=None):
    rclpy.init(args=args)
    camera_name = "top_camera"
    mediapipe_gesture_publisher = SixDRepnetHeadPosePublisher(
        node_name="sixdrepnet_head_pose_publisher",
        realsense_topic_name=f"/camera/{camera_name}/rgbd",
        annotated_image_topic_name="/sixdrepnet/head/annotated_image",
        head_pose_topic_name="/sixdrepnet/head/pose",
        source_frame_rel=f"{camera_name}_color_optical_frame",
        target_frame_rel="world",
        model_path=Path(
            "/home/ws/src/mediapipe_ros_pkg/models/6DRepNet_300W_LP_AFLW2000.pth"
        ),
        default_dtype=np.float32,
        face_detection_score_threshold=0.95,
        face_radius=0.09,
        gpu_id=-1,  # -1 for CPU, 0... for GPU
    )
    try:
        rclpy.spin(mediapipe_gesture_publisher)
    except KeyboardInterrupt:
        mediapipe_gesture_publisher.destroy_node()


class SixDRepnetHeadPosePublisher(RealsenseSubsctiber):
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
        face_detection_score_threshold,
        face_radius,
        gpu_id,
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

        self.default_dtype = default_dtype
        self.face_detection_score_threshold = face_detection_score_threshold
        self.face_radius = face_radius

        self.bridge = CvBridge()

        cudnn.enabled = True
        if gpu_id < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")

        self.model = SixDRepNet(
            backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.detector = RetinaFace(gpu_id=gpu_id)

        self.transformations = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def callback(self, rgbd_msg):
        rgb_image_msg = rgbd_msg.rgb
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")

        pose = self.detect(rgb_image, rgbd_msg.depth, rgbd_msg.rgb_camera_info)

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
            self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
        )

        return

    def detect(self, rgb_image, depth_image, rgb_cemara_info):
        with torch.no_grad():
            faces = self.detector(rgb_image)
            # TODO: To use only the first face
            box, landmarks, score = faces[0]

            # Print the location of each face in this image
            if score > self.face_detection_score_threshold:
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * bbox_height))
                y_min = max(0, y_min - int(0.2 * bbox_width))
                x_max = x_max + int(0.2 * bbox_height)
                y_max = y_max + int(0.2 * bbox_width)

                cv2.rectangle(
                    rgb_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 255, 255),
                    2,
                )

                img = rgb_image[y_min:y_max, x_min:x_max]
                img = PILImage.fromarray(img)
                img = img.convert("RGB")
                img = self.transformations(img)

                img = torch.Tensor(img[None, :]).to(self.device)

                start = time.time()
                R_pred = self.model(img)
                end = time.time()
                print("Head pose estimation: %2f ms" % ((end - start) * 1000.0))

                euler = (
                    self.compute_euler_angles_from_rotation_matrices(R_pred)
                    * 180
                    / np.pi
                )
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                # utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                self.plot_pose_cube(
                    rgb_image,
                    y_pred_deg,
                    p_pred_deg,
                    r_pred_deg,
                    x_min + int(0.5 * (x_max - x_min)),
                    y_min + int(0.5 * (y_max - y_min)),
                    size=bbox_width,
                )

                image_point = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                object_point = self.estimate_object_points(
                    image_point,
                    x_max - x_min,
                    y_max - y_min,
                    self.bridge.imgmsg_to_cv2(depth_image, "passthrough"),
                    rgb_cemara_info,
                )
                # TODO: Head is assumed the cylinder
                object_point = (
                    object_point[0],
                    object_point[1],
                    object_point[2] + self.face_radius,
                )

                if object_point is not None:
                    position = Point(
                        x=object_point[0],
                        y=object_point[1],
                        z=object_point[2],
                    )

                    orientation = Rotation.from_euler(
                        "xyz",
                        [
                            p_pred_deg.numpy()[0],
                            r_pred_deg.numpy()[0],
                            -y_pred_deg.numpy()[0] + 30,
                        ],
                        degrees=True,
                    ).as_quat()
                    # self.get_logger().info(
                    #     f"roll {r_pred_deg.numpy()[0]}, pitch {p_pred_deg.numpy()[0]}, yaw {y_pred_deg.numpy()[0]}"
                    # )
                    orientation = Quaternion(
                        x=orientation[0],
                        y=orientation[1],
                        z=orientation[2],
                        w=orientation[3],
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
                    from tf2_geometry_msgs import tf2_geometry_msgs

                    pose = tf2_geometry_msgs.do_transform_pose(pose, transform)

                    ###--------------------------------------
                    # TODO: To correct because the orientation does not match the ROS2 coordinate system
                    from mediapipe_ros_pkg.util import quaternion_multiply

                    rotation = Rotation.from_euler(
                        "xyz", [0, 0, np.pi / 2], degrees=False
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

            return None

    def estimate_object_points(
        self,
        image_point,  # center of the object
        width,
        height,
        depth_image,
        rgb_cemara_info,
        percentile=25,
    ):
        _intrinsics = intrinsics()
        _intrinsics.width = rgb_cemara_info.width
        _intrinsics.height = rgb_cemara_info.height
        _intrinsics.fx = rgb_cemara_info.k[0]
        _intrinsics.fy = rgb_cemara_info.k[4]
        _intrinsics.ppx = rgb_cemara_info.k[2]
        _intrinsics.ppy = rgb_cemara_info.k[5]

        cropped_depth = depth_image[
            max(0, image_point[1] - height // 2) : min(
                depth_image.shape[0] - 1, image_point[1] + height // 2 + 1
            ),
            max(0, image_point[0] - width // 2) : min(
                depth_image.shape[1] - 1, image_point[0] + width // 2 + 1
            ),
        ]

        # Filter out invalid depth values
        valid_depths = cropped_depth[cropped_depth > 0]

        # Crop around the pixel and get the depth value using the nearest 12.5% of the depth value
        # TODO: Fix hard code
        # breakpoint()
        if valid_depths.size > 0:
            depth = np.percentile(valid_depths, percentile)
            object_point = rs2_deproject_pixel_to_point(
                _intrinsics, image_point, depth * 0.001
            )
            return object_point
        else:
            return None

    def plot_pose_cube(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=150.0):
        # Input is a cv2 image
        # pose_params: (pitch, yaw, roll, tdx, tdy)
        # Where (tdx, tdy) is the translation of the face.
        # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

        p = pitch * np.pi / 180
        y = -(yaw * np.pi / 180)
        r = roll * np.pi / 180
        if tdx is not None and tdy is not None:
            face_x = tdx - 0.50 * size
            face_y = tdy - 0.50 * size

        else:
            height, width = img.shape[:2]
            face_x = width / 2 - 0.5 * size
            face_y = height / 2 - 0.5 * size

        x1 = size * (cos(y) * cos(r)) + face_x
        y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
        x2 = size * (-cos(y) * sin(r)) + face_x
        y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
        x3 = size * (sin(y)) + face_x
        y3 = size * (-cos(y) * sin(p)) + face_y

        # Draw base in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.line(
            img,
            (int(x2), int(y2)),
            (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
            (0, 0, 255),
            3,
        )
        cv2.line(
            img,
            (int(x1), int(y1)),
            (int(x1 + x2 - face_x), int(y1 + y2 - face_y)),
            (0, 0, 255),
            3,
        )
        # Draw pillars in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
        cv2.line(
            img,
            (int(x1), int(y1)),
            (int(x1 + x3 - face_x), int(y1 + y3 - face_y)),
            (255, 0, 0),
            2,
        )
        cv2.line(
            img,
            (int(x2), int(y2)),
            (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
            (255, 0, 0),
            2,
        )
        cv2.line(
            img,
            (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
            (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
            (255, 0, 0),
            2,
        )
        # Draw top in green
        cv2.line(
            img,
            (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
            (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            img,
            (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
            (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            img,
            (int(x3), int(y3)),
            (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            img,
            (int(x3), int(y3)),
            (int(x3 + x2 - face_x), int(y3 + y2 - face_y)),
            (0, 255, 0),
            2,
        )

        return img

    def compute_euler_angles_from_rotation_matrices(self, rotation_matrices):
        batch = rotation_matrices.shape[0]
        R = rotation_matrices
        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        singular = sy < 1e-6
        singular = singular.float()

        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
        ys = torch.atan2(-R[:, 2, 0], sy)
        zs = R[:, 1, 0] * 0

        gpu = rotation_matrices.get_device()
        if gpu < 0:
            out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(
                torch.device("cpu")
            )
        else:
            out_euler = torch.autograd.Variable(torch.zeros(batch, 3)).to(
                torch.device("cuda:%d" % gpu)
            )
        out_euler[:, 0] = x * (1 - singular) + xs * singular
        out_euler[:, 1] = y * (1 - singular) + ys * singular
        out_euler[:, 2] = z * (1 - singular) + zs * singular

        return out_euler
