"""Perception pipeline — sensor fusion and scene understanding.

The ``SensorFusion`` class bridges raw sensor data (from MuJoCo simulation
or ROS 2 hardware) into the LangGraph ``AgentGraphState``, replacing the
stub ``perceive_node`` with real multi-modal perception.
"""

from __future__ import annotations

from roboclaw.perception.sensor_fusion import SensorFusion

__all__ = ["SensorFusion"]
