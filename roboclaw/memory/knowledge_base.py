"""Knowledge base — structured domain facts for humanoid robotics.

Stores: object affordances, safety rules, kinematic limits, skill parameters.
Backed by PostgreSQL in production; in-memory dict fallback for development.
"""

from __future__ import annotations

from typing import Any


class KnowledgeBase:
    """Structured domain knowledge for the humanoid robot.

    Organizes facts by domain:
    - object_affordance: How objects can be grasped/manipulated
    - safety_rule: Hard safety constraints
    - kinematic_limit: Joint limits and workspace boundaries
    - skill_parameter: Default parameters for skills
    """

    def __init__(self, db_session_factory: Any = None) -> None:
        self._session_factory = db_session_factory
        # In-memory fallback
        self._entries: dict[str, dict[str, dict[str, Any]]] = {
            "object_affordance": {},
            "safety_rule": {},
            "kinematic_limit": {},
            "skill_parameter": {},
        }
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Seed the knowledge base with common humanoid robotics facts."""
        defaults = [
            # Object affordances
            ("object_affordance", "cup", {"grasp_type": "top_down_encompassing", "grip_force_n": 3.0, "approach_direction": "top", "pre_grasp_offset_m": 0.1}),
            ("object_affordance", "bottle", {"grasp_type": "cylindrical_encompassing", "grip_force_n": 5.0, "approach_direction": "side", "pre_grasp_offset_m": 0.08}),
            ("object_affordance", "door_handle", {"grasp_type": "hook_grasp", "grip_force_n": 10.0, "approach_direction": "front", "pre_grasp_offset_m": 0.05}),
            ("object_affordance", "box_small", {"grasp_type": "top_down_encompassing", "grip_force_n": 4.0, "approach_direction": "top", "pre_grasp_offset_m": 0.12}),
            ("object_affordance", "box_large", {"grasp_type": "bi_manual_pinch", "grip_force_n": 8.0, "approach_direction": "side", "pre_grasp_offset_m": 0.15, "requires_dual_arm": True}),
            ("object_affordance", "tray", {"grasp_type": "bi_manual_support", "grip_force_n": 2.0, "approach_direction": "bottom", "pre_grasp_offset_m": 0.1, "requires_dual_arm": True}),

            # Safety rules
            ("safety_rule", "max_gripper_force", {"limit_n": 15.0, "soft_limit_n": 12.0, "description": "Maximum force applied by gripper"}),
            ("safety_rule", "max_arm_velocity", {"limit_rad_s": 3.0, "soft_limit_rad_s": 2.5, "description": "Maximum arm joint velocity"}),
            ("safety_rule", "human_proximity_stop", {"distance_m": 0.5, "description": "E-stop if human closer than this distance"}),
            ("safety_rule", "zmp_margin", {"margin_m": 0.02, "description": "Minimum ZMP margin from support polygon edge"}),
            ("safety_rule", "max_payload_kg", {"limit_kg": 5.0, "description": "Maximum payload per arm"}),

            # Kinematic limits (general — per-embodiment overrides in robot config)
            ("kinematic_limit", "arm_workspace_radius", {"default_m": 0.85, "description": "Arm reach from shoulder"}),
            ("kinematic_limit", "gripper_max_opening", {"default_m": 0.12, "description": "Maximum gripper opening width"}),

            # Skill parameters
            ("skill_parameter", "whole_body_grasp", {"approach_speed_ms": 0.1, "grasp_timeout_s": 5.0, "post_grasp_retreat_m": 0.05}),
            ("skill_parameter", "walk_steps", {"step_height_m": 0.05, "step_duration_s": 0.6, "max_steps_per_plan": 20}),
            ("skill_parameter", "place_object", {"place_speed_ms": 0.05, "release_time_s": 0.3, "confirm_release": True}),
            ("skill_parameter", "handover", {"handover_height_m": 1.2, "handover_distance_m": 0.3, "grip_timeout_s": 3.0}),
        ]

        for domain, key, value in defaults:
            self._entries[domain][key] = value

    # --- CRUD ---

    async def query(self, domain: str, key: str) -> dict[str, Any] | None:
        """Query a specific knowledge entry."""
        if self._session_factory:
            try:
                async with self._session_factory() as session:
                    result = await session.execute(
                        "SELECT value_json FROM knowledge_entries WHERE domain = $1 AND key = $2",
                        domain, key,
                    )
                    row = result.fetchone()
                    if row:
                        return row["value_json"]
            except Exception:
                pass
        return self._entries.get(domain, {}).get(key)

    async def query_all(self, domain: str) -> list[dict[str, Any]]:
        """Get all entries in a domain."""
        if self._session_factory:
            try:
                async with self._session_factory() as session:
                    result = await session.execute(
                        "SELECT key, value_json FROM knowledge_entries WHERE domain = $1",
                        domain,
                    )
                    return [{"key": row["key"], **row["value_json"]} for row in result.fetchall()]
            except Exception:
                pass
        return [{"key": k, **v} for k, v in self._entries.get(domain, {}).items()]

    async def upsert(self, domain: str, key: str, value: dict[str, Any]) -> None:
        """Insert or update a knowledge entry."""
        self._entries.setdefault(domain, {})[key] = value
        if self._session_factory:
            try:
                import json
                async with self._session_factory() as session:
                    await session.execute(
                        """
                        INSERT INTO knowledge_entries (domain, key, value_json, updated_at)
                        VALUES ($1, $2, $3, now())
                        ON CONFLICT (domain, key) DO UPDATE SET value_json = $3, updated_at = now()
                        """,
                        domain, key, json.dumps(value),
                    )
                    await session.commit()
            except Exception:
                pass

    # --- Convenience ---

    async def get_affordance(self, object_label: str) -> dict[str, Any]:
        """Get manipulation affordances for an object type."""
        result = await self.query("object_affordance", object_label)
        return result or {"grasp_type": "top_down_encompassing", "grip_force_n": 3.0}

    async def get_safety_rules(self) -> list[dict[str, Any]]:
        """Get all safety rules."""
        return await self.query_all("safety_rule")

    async def get_skill_params(self, skill_name: str) -> dict[str, Any]:
        """Get default parameters for a skill."""
        result = await self.query("skill_parameter", skill_name)
        return result or {}

    async def close(self) -> None:
        """Clean up resources."""
        pass
