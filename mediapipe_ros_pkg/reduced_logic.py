import traceback as tb

import numpy as np
import rclpy

from crow_msgs.msg import CommandType
# from crow_robot_msgs.action import RobotAction
# from crow_robot_msgs.msg import (ActionResultFlag, ObjectType, RobotActionType, Units)

from trio3_ros2_interfaces.msg import (
    RobotStatus,
    ObjectType,
    CoreActionPhase,
    Units,
    GripperStatus,
    RobotActionType,
    ActionResultFlag,
)
from trio3_ros2_interfaces.srv import GetRobotStatus
from trio3_ros2_interfaces.action import RobotAction
from trio3_ros2_interfaces.msg import Units, RobotActionType, ObjectType
from trio3_ros2_interfaces.srv import GetMaskedPointCloud, GetGripPoints
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


from geometry_msgs.msg import Pose, Point
from rclpy.action import ActionClient
from rclpy.node import Node


class ControlLogic(Node):
    ROBOT_ACTION_POINT = "point"
    ROBOT_ACTION_FETCH = "pick_n_pass"

    PASS_TO_LOCATION = [0.550, 0.450, 0.270]

    PCL_GETTER_SERVICE_NAME = "get_masked_point_cloud_rs"
    GRIP_GETTER_SERVICE_NAME = "get_grip_points"

    def __init__(self, node_name="control_logic"):
        super().__init__(node_name)
        self.get_logger().info("Initializing ControlLogic node...")

        mtex_group = MutuallyExclusiveCallbackGroup()
        self.robot_point_client = ActionClient(
            self, RobotAction, self.ROBOT_ACTION_POINT, callback_group=mtex_group
        )
        self.robot_pass_client = ActionClient(
            self, RobotAction, self.ROBOT_ACTION_FETCH, callback_group=mtex_group
        )
        self.get_logger().info("Action clients initialized.")

        self._type_dict = {
            k: v
            for k, v in ObjectType.__dict__.items()
            if not k.startswith("_") and type(v) is int
        }
        self.COMMAND_DICT = {
            CommandType.POINT: self.sendPointAction,
            CommandType.FETCH: self.sendPassAction,
        }

        self.translate_action_type = {
            v: k
            for k, v in RobotActionType.__dict__.items()
            if not k.startswith("_") and type(v) is int
        }
        self.translate_result_flag = {
            v: k
            for k, v in ActionResultFlag.__dict__.items()
            if not k.startswith("_") and type(v) is int
        }

        self.srv_get_grip = self.create_service(
            GetGripPoints,
            self.GRIP_GETTER_SERVICE_NAME,
            self.get_grip,
            callback_group=ReentrantCallbackGroup(),
        )

    def sendPointAction(self, target_xyz, target_size):
        """Point: move to target and back home"""
        self.get_logger().info("Performing Point action")
        goal_msg = self.composeRobotActionMessage(
            target_xyz=target_xyz,
            target_size=target_size,
            action_type=RobotActionType.POINT,
        )

        self._send_goal_future = self.robot_point_client.send_goal_async(
            goal_msg, feedback_callback=self.robot_feedback_cb
        )
        self._send_goal_future.add_done_callback(self.robot_response_cb)

    def sendPassAction(self, target_xyz, target_size):
        """Pass (give): move to target, pick, move to user"""
        self.get_logger().info("Performing Pass action")
        goal_msg = self.composeRobotActionMessage(
            target_xyz=target_xyz,
            target_size=target_size,
            location_xyz=self.PASS_TO_LOCATION,
            action_type=RobotActionType.PICK_N_PASS,
        )

        self._send_goal_future = self.robot_pass_client.send_goal_async(
            goal_msg, feedback_callback=self.robot_feedback_cb
        )
        self._send_goal_future.add_done_callback(self.robot_response_cb)

    def get_grip(self, request, response):
        response.response_units.unit_type = Units.METERS

        request_pose = np.r_[
            "0,1,0", [getattr(request.expected_position.position, a) for a in "xyz"]
        ]

        response.grip_pose = Pose()
        response.grip_pose.position = Point(
            **{k: v for k, v in zip("xyz", request_pose)}
        )
        response.success = True
        #
        return response

    def composeRobotActionMessage(
        self,
        target_xyz=None,
        target_size=None,
        target_type=None,
        location_xyz=None,
        action_type=-1,
        robot_id=-1,
    ):
        """Composes a standard robot goal message.

        Args:
            target_xyz (list[float], optional): XYZ coordinates of the target object (to pick-up). Defaults to None.
            target_size (list[float], optional): Axis aligned size of the target object in meters. Defaults to None.
            target_type (int, optional): Object type from the ObjectType enum. Defaults to None.
            location_xyz (list[float], optional): XYZ coordinates of the location to place the object. Defaults to None.
            action_type (int, optional): Action type from the RobotActionType enum. Defaults to -1.
            robot_id (int, optional): Robot ID. Defaults to -1.
        """
        goal_msg = RobotAction.Goal()
        goal_msg.frame_id = "global"
        goal_msg.robot_id = robot_id
        goal_msg.request_units = Units(unit_type=Units.METERS)
        goal_msg.robot_action_type.type = action_type

        goal_msg.poses = []
        if target_xyz is not None:
            pick_pose = Pose()
            pick_pose.position.x, pick_pose.position.y, pick_pose.position.z = (
                target_xyz
            )
            goal_msg.poses.append(pick_pose)
            if target_size is None:  # or target_type==-1:
                goal_msg.size = [0.0, 0.0, 0.0]
            else:
                goal_msg.size = target_size
            if target_type is None:
                goal_msg.object_type.type = -1
            else:
                goal_msg.object_type = ObjectType(type=target_type)

        if np.isnan(goal_msg.size[0]):
            goal_msg.size = [0, 0, 0]
        if location_xyz is not None:
            place_pose = Pose()
            place_pose.position.x, place_pose.position.y, place_pose.position.z = (
                location_xyz
            )
            goal_msg.poses.append(place_pose)

        self.get_logger().info(
            f"Composed a goal message for {self.translate_action_type[goal_msg.robot_action_type.type]}."
        )
        return goal_msg

    def robot_response_cb(self, future):
        """This function is called when the robot responses to the goal request.
        It can either accept or reject the goal.
        """
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error("Goal rejected :(")
            return

        self.get_logger().info("Goal accepted :)")

        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robot_done_cb)

    def robot_done_cb(self, future):
        """This function is called when the robot is done executing the set goal.
        It can be when the action is completed or failed (future.result().done == False).
        """
        result = future.result().result
        if result.done:
            self.get_logger().info("Action done.")
        else:
            self.get_logger().error(
                f"Action failed because: {result.action_result_flag} [{self.translate_result_flag[result.action_result_flag.flag]}]"
            )
        self.get_logger().info("Robot action done.")

    def robot_feedback_cb(self, feedback_msg):
        """This function is called repeatedly during execution of the goal by the robot."""
        feedback = feedback_msg.feedback
        self.get_logger().info(f"RobotAction feedback: {feedback.status}")

    def cancel_current_goal(self, wait=False):
        """Requests cancellation of the current goal.

        Returns:
            bool: Returns True, if goal can be canceled
        """
        self.get_logger().info("Trying to cancel the current goal...")
        if self.goal_handle is None:
            self.get_logger().warn("Goal handle is None, cannot cancel.")
            return False
        if wait:
            response = self.goal_handle.cancel_goal()
            self.get_logger().info(f"Response was: {response}")
        else:
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self.robot_canceling_done)
        return True

    def robot_canceling_done(self, future):
        """This function is called when the robot cancels or rejects to cancel the current action."""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("Goal successfully canceled")
        else:
            self.get_logger().info("Goal failed to cancel")


def main():
    rclpy.init()
    cl = ControlLogic()
    cl.get_logger().info("Init done.")
    try:
        mte = MultiThreadedExecutor(4)
        # cl.sendPointAction([0.2, 0.2, 0.02], [0.1, 0.1, 0.1])
        cl.sendPassAction([0.5, 0.0, 0.02], [0.1, 0.1, 0.1])
        rclpy.spin(cl, executor=mte)
        rclpy.spin_once(cl, executor=mte)
        cl.get_logger().info("ready")
    except KeyboardInterrupt:
        print("User requested shutdown.")
    except BaseException as e:
        print(f"Some error had occured: {e}")
        tb.print_exc()

    cl.destroy_node()


if __name__ == "__main__":
    main()
