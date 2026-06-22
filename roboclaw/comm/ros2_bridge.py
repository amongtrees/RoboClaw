"""ROS 2 bridge stub — topic/service adapter for robot hardware.

In production, this provides adapters for:
- Sensor topics (camera, LiDAR, joint states, IMU, force-torque)
- Action servers (locomotion, manipulation, navigation)
- TF2 transforms
- Robot description (URDF)

Phase 1 stub — logs expected ROS 2 interfaces.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ROS2Bridge:
    """Adapter between RoboClaw and ROS 2.

    Phase 1 stub. Requires ROS 2 installation (rclpy) for production use.
    """

    def __init__(self, robot_config: Any = None) -> None:
        self._config = robot_config
        self._initialized = False
        self._subscribers: dict[str, Any] = {}
        self._publishers: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize ROS 2 node and create subscribers/publishers."""
        try:
            import rclpy  # noqa: F401

            logger.info("ROS 2 bridge initialized")
            self._initialized = True
        except ImportError:
            logger.info("ROS 2 not installed — bridge in stub mode")
            self._initialized = False

    async def subscribe_sensor(self, topic_name: str, topic_type: str, callback: Any) -> None:
        """Subscribe to a ROS 2 sensor topic.

        Production topics:
        - /camera/color/image_raw (sensor_msgs/Image)
        - /camera/depth/image_rect_raw (sensor_msgs/Image)
        - /joint_states (sensor_msgs/JointState)
        - /imu/data (sensor_msgs/Imu)
        - /left_wrist/ft (geometry_msgs/WrenchStamped)
        - /right_wrist/ft (geometry_msgs/WrenchStamped)
        - /lidar/points (sensor_msgs/PointCloud2)
        """
        if self._initialized:
            logger.info(f"Subscribed to ROS 2 topic: {topic_name} [{topic_type}]")
        else:
            logger.debug(f"[STUB] Would subscribe to ROS 2 topic: {topic_name} [{topic_type}]")

    async def publish_command(self, topic_name: str, topic_type: str, message: Any) -> None:
        """Publish a command on a ROS 2 topic.

        Production topics:
        - /joint_trajectory_command (trajectory_msgs/JointTrajectory)
        - /gripper/left/command (std_msgs/Float64)
        - /gripper/right/command (std_msgs/Float64)
        """
        if self._initialized:
            logger.info(f"Published to ROS 2 topic: {topic_name}")
        else:
            logger.debug(f"[STUB] Would publish to ROS 2 topic: {topic_name}")

    async def call_action(self, action_name: str, action_type: str, goal: Any) -> Any:
        """Call a ROS 2 action server.

        Production actions:
        - /walk_to_pose (WalkToPose)
        - /whole_body_grasp (WholeBodyGrasp)
        - /navigate_to_room (NavigateToRoom)
        """
        if self._initialized:
            logger.info(f"Called ROS 2 action: {action_name}")
        else:
            logger.debug(f"[STUB] Would call ROS 2 action: {action_name}")
        return {"status": "simulated_success"}

    async def get_tf_transform(self, target_frame: str, source_frame: str) -> dict[str, Any]:
        """Get a TF2 transform between two frames."""
        # Stub
        return {
            "translation": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }

    async def shutdown(self) -> None:
        """Shutdown ROS 2 node."""
        self._initialized = False
        logger.info("ROS 2 bridge shutdown")
