"""Bridge from MuJoCo simulation to ROS 2 topics.

``MujocoROSPublisher`` converts MuJoCo snapshot dicts into standard
ROS 2 messages and publishes them via a ``ROS2Bridge``.  This allows
a MuJoCo-simulated robot to appear as a real robot on the ROS 2 bus:
it publishes ``/joint_states``, ``/imu/data``, and FT sensor topics
just like real hardware would.

Usage inside a MuJoCo executor::

    ros_pub = MujocoROSPublisher(bridge, joint_names)
    async for ctrl in trajectory:
        snap = await env.step(ctrl)
        ros_pub.publish_snapshot(snap)  # → /joint_states, /imu/data
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class MujocoROSPublisher:
    """Converts MuJoCo snapshots to ROS 2 sensor messages and publishes them.

    Parameters
    ----------
    bridge:
        A ``ROS2Bridge`` instance (must be initialized with rclpy available).
    joint_names:
        Ordered list of joint names matching the MuJoCo model.
    node:
        Optional pre-existing rclpy Node.  If omitted, messages are
        built and published through ``bridge.publish_command``.
    """

    def __init__(
        self,
        bridge: Any,
        joint_names: list[str] | None = None,
    ) -> None:
        self._bridge = bridge
        self._joint_names = joint_names or []
        self._seq: int = 0

        # Direct ROS publishers (populated if bridge is live)
        self._joint_state_pub: Any = None
        self._imu_pub: Any = None
        self._ft_left_pub: Any = None
        self._ft_right_pub: Any = None

        self._setup_publishers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Convert a MuJoCo snapshot to ROS 2 messages and publish.

        Called after each ``env.step()`` in the asyncio thread.
        The actual ROS publishing is thread-safe via rclpy's publisher.
        """
        if not snapshot:
            return

        self._seq += 1
        now = time.time()

        sec = int(now)
        nsec = int((now - sec) * 1e9)

        # --- JointState ---
        self._publish_joint_state(snapshot, sec, nsec)

        # --- IMU ---
        self._publish_imu(snapshot, sec, nsec)

    # ------------------------------------------------------------------
    # Internal publishers
    # ------------------------------------------------------------------

    def _publish_joint_state(
        self, snapshot: dict[str, Any], sec: int, nsec: int
    ) -> None:
        """Build and publish ``sensor_msgs/JointState``."""
        joint_positions = snapshot.get("joint_positions", {})
        joint_velocities = snapshot.get("joint_velocities", {})

        if not joint_positions:
            return

        names: list[str] = []
        positions: list[float] = []
        velocities: list[float] = []

        for name in self._joint_names:
            if name in joint_positions:
                names.append(name)
                positions.append(float(joint_positions[name]))
                velocities.append(float(joint_velocities.get(name, 0.0)))

        if not names:
            return

        if self._joint_state_pub is not None:
            import sensor_msgs.msg

            msg = sensor_msgs.msg.JointState()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
            msg.header.frame_id = "world"
            msg.name = names
            msg.position = positions
            msg.velocity = velocities if any(v != 0.0 for v in velocities) else []
            self._joint_state_pub.publish(msg)
        else:
            logger.debug("[STUB] JointState: %d joints", len(names))

    def _publish_imu(self, snapshot: dict[str, Any], sec: int, nsec: int) -> None:
        """Build and publish ``sensor_msgs/Imu``."""
        imu = snapshot.get("imu", {})
        if not imu:
            return

        accel = imu.get("accelerometer", [0.0, 0.0, -9.81])
        gyro = imu.get("gyroscope", [0.0, 0.0, 0.0])

        if self._imu_pub is not None:
            import sensor_msgs.msg

            msg = sensor_msgs.msg.Imu()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
            msg.header.frame_id = "imu_link"
            msg.linear_acceleration.x = float(accel[0])
            msg.linear_acceleration.y = float(accel[1])
            msg.linear_acceleration.z = float(accel[2])
            msg.angular_velocity.x = float(gyro[0])
            msg.angular_velocity.y = float(gyro[1])
            msg.angular_velocity.z = float(gyro[2])
            # Covariance: -1 = unknown
            msg.linear_acceleration_covariance[0] = -1.0
            msg.angular_velocity_covariance[0] = -1.0
            self._imu_pub.publish(msg)
        else:
            logger.debug("[STUB] IMU: accel=%s", accel)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_publishers(self) -> None:
        """Create ROS 2 publishers if the bridge has a live node."""
        if not self._bridge.is_initialized:
            logger.info("ROS 2 bridge not initialized — MuJoCo publisher in stub mode")
            return

        try:
            node = self._bridge._node
            if node is None:
                return

            import sensor_msgs.msg

            ros2_cfg = self._bridge._get_ros2_config()
            js_topic = ros2_cfg.get("joint_state_topic", "/joint_states")
            imu_topic = ros2_cfg.get("imu_topic", "/imu/data")
            left_ft_topic = ros2_cfg.get("left_ft_topic", "/left_wrist/ft")
            right_ft_topic = ros2_cfg.get("right_ft_topic", "/right_wrist/ft")

            self._joint_state_pub = node.create_publisher(
                sensor_msgs.msg.JointState, js_topic, 10
            )
            self._imu_pub = node.create_publisher(
                sensor_msgs.msg.Imu, imu_topic, 10
            )

            # FT publishers (WrenchStamped)
            try:
                import geometry_msgs.msg
                self._ft_left_pub = node.create_publisher(
                    geometry_msgs.msg.WrenchStamped, left_ft_topic, 10
                )
                self._ft_right_pub = node.create_publisher(
                    geometry_msgs.msg.WrenchStamped, right_ft_topic, 10
                )
            except Exception:
                logger.debug("FT publishers not created")

            logger.info(
                "MujocoROSPublisher live: joint_state→%s, imu→%s",
                js_topic, imu_topic,
            )
        except Exception as exc:
            logger.warning("Failed to create ROS publishers: %s", exc)
