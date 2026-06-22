"""Pydantic Settings for all infrastructure and per-embodiment robot configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "roboclaw"
    user: str = "roboclaw"
    password: str = "roboclaw_dev"
    pool_min_size: int = 5
    pool_max_size: int = 20

    @property
    def dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    socket_timeout: float = 5.0

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class MilvusCollectionConfig(BaseModel):
    name: str
    dim: int = 1536
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"


class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    db_name: str = "roboclaw"
    collections: dict[str, MilvusCollectionConfig] = Field(default_factory=dict)


class RabbitMQConfig(BaseModel):
    host: str = "localhost"
    port: int = 5672
    user: str = "roboclaw"
    password: str = "roboclaw_dev"
    vhost: str = "/"
    exchanges: dict[str, str] = Field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"amqp://{self.user}:{self.password}@{self.host}:{self.port}/{self.vhost}"


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 4096
    planning: "LLMConfig | None" = None


class AgentSettings(BaseModel):
    loop_rate_hz: int = 50
    perception_rate_hz: int = 10
    planning_min_interval_s: float = 1.0
    max_recovery_attempts: int = 3
    default_timeout_s: float = 30.0
    safety_check_rate_hz: int = 100


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60


class A2AConfig(BaseModel):
    enabled: bool = True
    discovery_interval_s: float = 30.0
    task_timeout_s: float = 300.0


class EmbodimentConfig(BaseModel):
    num_arms: int = 2
    num_legs: int = 2
    num_fingers_per_hand: int = 5
    height_m: float = 1.80
    mass_kg: float = 47.0
    joint_limits: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    force_limits: dict[str, float] = Field(default_factory=dict)


class SafetyBounds(BaseModel):
    zmp_support_polygon: list[list[float]] = Field(default_factory=list)
    max_joint_velocity_rad_s: float = 5.0
    collision_force_threshold_n: float = 50.0
    estop_deceleration: float = 2.0


class SkillDefaults(BaseModel):
    default_grasp_strategy: str = "top_down_encompassing"
    walk_max_speed_ms: float = 1.5
    walk_step_height_m: float = 0.05
    arm_workspace_radius_m: float = 0.85


class RobotConfig(BaseModel):
    """Per-embodiment robot configuration loaded from YAML."""

    model: str = "h1_unitree"
    display_name: str = ""
    description: str = ""
    embodiment: EmbodimentConfig = Field(default_factory=EmbodimentConfig)
    safety: SafetyBounds = Field(default_factory=SafetyBounds)
    skills: SkillDefaults = Field(default_factory=SkillDefaults)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotConfig":
        """Load robot config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        robot_data = raw.get("robot", raw)
        return cls(
            model=robot_data.get("model", ""),
            display_name=robot_data.get("display_name", ""),
            description=robot_data.get("description", ""),
            embodiment=EmbodimentConfig(**raw.get("embodiment", {})),
            safety=SafetyBounds(**raw.get("safety", {})),
            skills=SkillDefaults(**raw.get("skills", {})),
        )


class Settings(BaseSettings):
    """Top-level application settings, loaded from YAML config and env vars."""

    model_config = SettingsConfigDict(env_prefix="ROBOCLAW_", env_nested_delimiter="__")

    robot: RobotConfig = Field(default_factory=RobotConfig)
    infrastructure: dict[str, Any] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APIConfig = Field(default_factory=APIConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path, robot_config_path: str | Path | None = None) -> "Settings":
        """Load settings from a YAML configuration file and optional robot-specific config."""
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        infra = raw.get("infrastructure", {})
        postgres = PostgresConfig(**infra.get("postgres", {}))
        redis = RedisConfig(**infra.get("redis", {}))
        milvus_raw = infra.get("milvus", {})
        milvus_collections = {
            k: MilvusCollectionConfig(**v)
            for k, v in milvus_raw.get("collections", {}).items()
        }
        milvus = MilvusConfig(
            host=milvus_raw.get("host", "localhost"),
            port=milvus_raw.get("port", 19530),
            db_name=milvus_raw.get("db_name", "roboclaw"),
            collections=milvus_collections,
        )
        rabbitmq_raw = infra.get("rabbitmq", {})
        rabbitmq = RabbitMQConfig(
            host=rabbitmq_raw.get("host", "localhost"),
            port=rabbitmq_raw.get("port", 5672),
            user=rabbitmq_raw.get("user", "roboclaw"),
            password=rabbitmq_raw.get("password", "roboclaw_dev"),
            vhost=rabbitmq_raw.get("vhost", "/"),
            exchanges=rabbitmq_raw.get("exchanges", {}),
        )

        llm_raw = raw.get("llm", {})
        planning_llm = llm_raw.get("planning", {})
        llm = LLMConfig(
            provider=llm_raw.get("provider", "anthropic"),
            model=llm_raw.get("model", "claude-sonnet-4-6"),
            temperature=llm_raw.get("temperature", 0.1),
            max_tokens=llm_raw.get("max_tokens", 4096),
            planning=LLMConfig(
                provider=planning_llm.get("provider", llm_raw.get("provider", "anthropic")),
                model=planning_llm.get("model", llm_raw.get("model", "claude-sonnet-4-6")),
                temperature=planning_llm.get("temperature", 0.1),
                max_tokens=planning_llm.get("max_tokens", 8192),
            ) if planning_llm else None,
        )

        robot_config = RobotConfig()
        if robot_config_path:
            robot_config = RobotConfig.from_yaml(robot_config_path)

        return cls(
            robot=robot_config,
            infrastructure={
                "postgres": postgres,
                "redis": redis,
                "milvus": milvus,
                "rabbitmq": rabbitmq,
            },
            llm=llm,
            agent=AgentSettings(**raw.get("agent", {})),
            api=APIConfig(**raw.get("api", {})),
            a2a=A2AConfig(**raw.get("a2a", {})),
        )

    @property
    def postgres(self) -> PostgresConfig:
        return self.infrastructure["postgres"]

    @property
    def redis(self) -> RedisConfig:
        return self.infrastructure["redis"]

    @property
    def milvus(self) -> MilvusConfig:
        return self.infrastructure["milvus"]

    @property
    def rabbitmq(self) -> RabbitMQConfig:
        return self.infrastructure["rabbitmq"]
