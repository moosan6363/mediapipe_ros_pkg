from ultralytics import YOLO

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

from mediapipe_ros_pkg.realsense_subscriber import RealsenseSubsctiber
import cv2
from pyrealsense2 import intrinsics, rs2_deproject_pixel_to_point
import numpy as np


def main(args=None):
    rclpy.init(args=args)
    camera_name = "left_camera"
    mediapipe_objectron_publisher = YOLOObjectDetectionPublisher(
        node_name="mediapipe_objectron_publisher",
        realsense_topic_name=f"/camera/{camera_name}/rgbd",
        annotated_image_topic_name="/yolo/object_detection/annotated_image",
        objectron_marker_array_topic_name="/yolo/object_detection/marker_array",
        source_frame_rel=f"{camera_name}_color_optical_frame",
        target_frame_rel="world",
        cup_radius=0.06,
        cup_height=0.08,
    )
    try:
        rclpy.spin(mediapipe_objectron_publisher)
    except KeyboardInterrupt:
        mediapipe_objectron_publisher.destroy_node()


class YOLOObjectDetectionPublisher(RealsenseSubsctiber):
    def __init__(
        self,
        node_name,
        realsense_topic_name,
        annotated_image_topic_name,
        objectron_marker_array_topic_name,
        source_frame_rel,
        target_frame_rel,
        cup_radius,
        cup_height,
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

        self.bridge = CvBridge()

        self.model = YOLO("yolo11n.pt")

        self.cup_radius = cup_radius
        self.cup_height = cup_height

    def callback(self, rgbd_msg):
        image = self.bridge.imgmsg_to_cv2(rgbd_msg.rgb, "rgb8")

        results = self.model(image)
        cups = []

        annotated_image = image.copy()
        for result in results:
            for box in result.boxes:
                cls = box.cls.cpu().numpy()[0]
                cls_name = result.names[int(cls)]
                confidence = box.conf.cpu().numpy()[0]
                label = f"{cls_name} {confidence:.2f}"
                xywh = box.xywh.cpu().numpy()[0]
                x, y, w, h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])

                if cls_name == "cup":
                    cups.append((x, y, w, h))

                cv2.rectangle(
                    annotated_image,
                    (x - w // 2, y - h // 2),
                    (x + w // 2, y + h // 2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (x - w // 2, y - h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        self.annotated_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
        )

        self.create_marker(rgbd_msg, cups)
        return

    def create_marker(self, rgbd_msg, cups):
        markers = []
        for id, cup in enumerate(cups):
            image_point = (cup[0], cup[1])
            object_point = self.estimate_object_points(
                image_point,
                cup[2],
                cup[3],
                self.bridge.imgmsg_to_cv2(rgbd_msg.depth, "passthrough"),
                rgbd_msg.rgb_camera_info,
            )

            if object_point is not None:
                # TODO: Head is assumed the cylinder
                object_point = (
                    object_point[0],
                    object_point[1],
                    object_point[2] + self.cup_radius - 0.04,
                )

                point_stamped = PointStamped(
                    header=Header(frame_id=self.source_frame_rel),
                    point=Point(
                        x=object_point[0],
                        y=object_point[1],
                        z=object_point[2],
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
                        position=point_stamped.point,
                        orientation=Quaternion(),
                    ),
                    scale=Vector3(
                        x=self.cup_radius * 2,
                        y=self.cup_radius * 2,
                        z=self.cup_height,
                    ),
                    color=ColorRGBA(
                        r=1.0,
                        g=0.0,
                        b=0.0,
                        a=0.5,
                    ),
                )
                markers.append(marker)

        if len(markers) > 0:
            self.objectron_marker_array_publisher.publish(MarkerArray(markers=markers))

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
