# 核心层 (Core Layer)

核心层是整个框架的基础，提供配置管理、类型定义、错误体系和依赖注入容器。位于 `roboclaw/core/`。

---

## 一、配置系统 (`config.py`)

使用 **Pydantic Settings** 和 **YAML 配置** 驱动整个框架的参数管理。

### 1.1 基础设施配置

```python
# PostgreSQL 连接配置
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
        # 自动组装 asyncpg 连接字符串
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# Redis 连接配置
class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    socket_timeout: float = 5.0

# Milvus 向量数据库配置（支持多 Collection）
class MilvusCollectionConfig(BaseModel):
    name: str
    dim: int = 1536        # 向量维度（默认 OpenAI embedding 维度）
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"

class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    db_name: str = "roboclaw"
    collections: dict[str, MilvusCollectionConfig] = {}

# RabbitMQ 消息队列配置
class RabbitMQConfig(BaseModel):
    host: str = "localhost"
    port: int = 5672
    vhost: str = "/"
    exchanges: dict[str, str] = {}  # Exchange 名称到类型映射
```

### 1.2 应用配置

```python
class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 4096
    planning: "LLMConfig | None" = None  # 规划专用模型（可用更强的模型）

class AgentSettings(BaseModel):
    loop_rate_hz: int = 50           # 主循环频率
    perception_rate_hz: int = 10     # 感知频率
    planning_min_interval_s: float = 1.0  # 最小规划间隔
    max_recovery_attempts: int = 3   # 最大恢复尝试次数
    safety_check_rate_hz: int = 100  # 安全检查频率

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 60

class A2AConfig(BaseModel):
    enabled: bool = True
    discovery_interval_s: float = 30.0
    task_timeout_s: float = 300.0
```

### 1.3 机器人形态配置

```python
class EmbodimentConfig(BaseModel):
    num_arms: int = 2
    num_legs: int = 2
    num_fingers_per_hand: int = 5
    height_m: float = 1.80
    mass_kg: float = 47.0
    joint_limits: dict[str, dict[str, list[float]]] = {}  # 关节限位
    force_limits: dict[str, float] = {}                    # 力/力矩限制

class SafetyBounds(BaseModel):
    zmp_support_polygon: list[list[float]] = []  # ZMP 支撑多边形
    max_joint_velocity_rad_s: float = 5.0
    collision_force_threshold_n: float = 50.0
    estop_deceleration: float = 2.0

class SkillDefaults(BaseModel):
    default_grasp_strategy: str = "top_down_encompassing"
    walk_max_speed_ms: float = 1.5
    walk_step_height_m: float = 0.05
    arm_workspace_radius_m: float = 0.85
```

### 1.4 顶层配置与加载

```python
class Settings(BaseSettings):
    """从 YAML 文件和 ROBOCLAW_ 前缀环境变量加载"""
    model_config = SettingsConfigDict(env_prefix="ROBOCLAW_", env_nested_delimiter="__")

    robot: RobotConfig
    infrastructure: dict[str, Any]  # 内含 PostgresConfig / RedisConfig 等
    llm: LLMConfig
    agent: AgentSettings
    api: APIConfig
    a2a: A2AConfig

    @classmethod
    def from_yaml(cls, config_path, robot_config_path=None) -> "Settings":
        # 1. 解析主配置文件
        # 2. 加载机器人特定配置
        # 3. 组装所有子配置对象
        # 4. 返回完整 Settings 实例

    @property
    def postgres(self) -> PostgresConfig: ...
    @property
    def redis(self) -> RedisConfig: ...
    @property
    def milvus(self) -> MilvusConfig: ...
    @property
    def rabbitmq(self) -> RabbitMQConfig: ...
```

**关键设计**：环境变量覆盖 — 通过 `ROBOCLAW_` 前缀的环境变量可以覆盖任何 YAML 配置，例如 `ROBOCLAW_LLM__MODEL=claude-opus-4-8` 可以覆盖模型选择。

---

## 二、类型枚举 (`types.py`)

集中管理所有领域枚举，使用 Python 3.11 的 `StrEnum`：

### AgentPhase — Agent 循环的阶段
```python
class AgentPhase(StrEnum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    ACTING = "acting"
    EVALUATING = "evaluating"    # 对应 Feedback 阶段
    RECOVERING = "recovering"
    DELEGATING = "delegating"    # A2A 委托
    HRI = "hri"                  # 人机交互
```

### Outcome — 执行结果
```python
class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
```

### SkillType — 技能目录（14种）
```python
class SkillType(StrEnum):
    NAVIGATE_TO = "navigate_to"           # 导航到指定房间
    WALK_STEPS = "walk_steps"             # 走指定步数
    WHOLE_BODY_GRASP = "whole_body_grasp" # 全身协调抓取
    BI_MANUAL_CARRY = "bi_manual_carry"   # 双臂搬运
    PLACE_OBJECT = "place_object"         # 放置物体
    HANDOVER = "handover"                 # 物体交接
    SPEAK = "speak"                       # 语音输出
    GAZE_AT = "gaze_at"                   # 注视目标
    WAIT_FOR = "wait_for"                 # 等待条件
    OPEN_DOOR = "open_door"               # 开门
    CLIMB_STAIRS = "climb_stairs"         # 爬楼梯
    PUSH_OBJECT = "push_object"           # 推物体
    PULL_OBJECT = "pull_object"           # 拉物体
    PRESS_BUTTON = "press_button"         # 按按钮
```

### FailureType — 故障分类
```python
class FailureType(StrEnum):
    KINEMATIC = "kinematic"         # 运动学故障（IK无解、关节超限）
    DYNAMIC = "dynamic"             # 动力学故障（力矩不足、振动）
    PERCEPTION = "perception"       # 感知故障（目标丢失、传感器超时）
    PLANNING = "planning"           # 规划故障（无可行规划、超时）
    SAFETY = "safety"               # 安全故障（碰撞、ZMP越界）
    COMMUNICATION = "communication" # 通信故障（ROS节点断开、消息丢失）
    HARDWARE = "hardware"           # 硬件故障（电机过热、电源异常）
    UNKNOWN = "unknown"             # 未知故障
```

### AgentMode — Agent 工作模式
```python
class AgentMode(StrEnum):
    IDLE = "idle"
    AUTONOMOUS = "autonomous"       # 全自主模式
    TELEOP = "teleop"               # 遥操作模式
    DIAGNOSTIC = "diagnostic"       # 诊断模式
    EMERGENCY_STOP = "emergency_stop" # 急停
```

### SafetySeverity — 安全事件等级
```python
class SafetySeverity(StrEnum):
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    CRITICAL = "critical"   # 严重
    E_STOP = "e_stop"       # 急停
```

---

## 三、领域错误体系 (`errors.py`)

采用分层继承的错误体系，每个错误都携带结构化信息：

```
RoboClawError (基类)
├── PerceptionError          # 感知错误
│   ├── SensorTimeoutError   #   传感器超时
│   └── ObjectNotFoundError  #   目标物体未找到
├── PlanningError            # 规划错误
│   ├── NoFeasiblePlanError  #   无可行规划
│   └── PlanValidationError  #   规划验证失败
├── ActionError              # 执行错误
│   ├── KinematicError       #   运动学错误（关节超限）
│   ├── GraspFailedError     #   抓取失败
│   └── CollisionError       #   意外碰撞
├── SafetyViolationError     # 安全违反
│   ├── ZMPViolationError    #   ZMP 超出支撑多边形
│   └── ForceLimitExceededError # 力/力矩超限
├── CommunicationError       # 通信错误
│   └── ROS2NodeError        #   ROS 2 节点故障
├── MemoryError              # 记忆系统错误
│   └── MemoryStoreError     #   记忆存储故障
└── A2AError                 # A2A 协议错误
    └── AgentUnavailableError #   同伴不可达
```

**关键设计**：
- 所有错误继承自 `RoboClawError`，携带 `code`（机器可读错误码）、`detail`（结构化上下文字典）
- `to_dict()` 方法用于 API 响应序列化
- 每个子类在构造时自动填充 `code`，消除了字符串拼写错误

---

## 四、依赖注入容器 (`registry.py`)

一个轻量级的 IoC（控制反转）容器，用于框架内部组件装配：

```python
class ComponentRegistry:
    def register(self, interface: type, implementation: Any) -> None:
        """注册单例实现"""
    def register_factory(self, interface: type, factory: Any) -> None:
        """注册工厂函数（每次 get 时调用）"""
    def register_named(self, name: str, component: Any) -> None:
        """按名称注册组件"""
    def get(self, interface: type) -> Any:
        """获取实现（优先返回单例，否则调用工厂）"""
    def get_named(self, name: str) -> Any:
        """按名称获取组件"""
    def has(self, interface: type) -> bool:
        """检查接口是否已注册"""
    def reset(self) -> None:
        """重置（测试用）"""
```

**使用模式**：
```python
# 注册阶段
registry = get_registry()
registry.register(AbstractPlanner, TaskDecomposer(llm=my_llm))
registry.register(EpisodicMemory, episodic_memory_instance)

# 使用阶段
planner = registry.get(AbstractPlanner)
episodic = registry.get_named("episodic_memory")
```

提供全局单例 `get_registry()` 和 `reset_registry()`（测试隔离用）。
