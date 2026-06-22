# 数据模型层 (Pydantic v2 Models)

所有数据模型使用 Pydantic v2 实现，提供自动校验、序列化和类型安全。位于 `roboclaw/models/`，包含 10 个模块。

---

## 一、核心状态模型 (`state.py` + `pose.py`)

### 1.1 Pose — 基础位姿

```python
class Pose(BaseModel):
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # 四元数 xyzw
```

### 1.2 RobotState — 通用机器人状态（跨形态基类）

```python
class RobotState(BaseModel):
    # 身份标识
    robot_id: str
    timestamp: float          # 自动填充当前时间戳
    frame_id: str = "base_link"
    task_id: str | None = None
    skill_name: str | None = None
    phase: str | None = None

    # 物理状态
    joint_states: dict[str, float] = {}      # 关节名 → 位置(rad)
    end_effector_pose: Pose | None = None    # 末端执行器位姿
    gripper_state: dict[str, Any] = {}       # 夹爪状态

    # 感知状态
    force_torque: ForceTorque | None = None  # 力/力矩传感器
    visible_objects: list[dict] = []         # 可见物体列表
    safety_state: dict = {}                  # 安全状态
    tf_snapshot: dict = {}                   # TF 变换快照
    metadata: dict = {}
```

### 1.3 HumanoidState — 人形机器人状态（继承 RobotState）

这是框架的核心状态模型，扩展了双足机器人和双臂操作特有的状态字段：

```python
class HumanoidState(RobotState):
    # ===== 双臂状态 =====
    end_effector_pose_left: Pose | None = None   # 左机械臂末端位姿
    end_effector_pose_right: Pose | None = None  # 右机械臂末端位姿
    gripper_state_left: dict = {}
    gripper_state_right: dict = {}
    force_torque_left: ForceTorque | None = None  # 左臂力传感器
    force_torque_right: ForceTorque | None = None # 右臂力传感器

    # ===== 双足平衡核心指标 =====
    zmp: tuple[float, float] = (0.0, 0.0)        # 零力矩点 (ZMP)
    cop_left: tuple[float, float] = (0.0, 0.0)   # 左脚压力中心
    cop_right: tuple[float, float] = (0.0, 0.0)  # 右脚压力中心
    left_foot_pose: Pose | None = None            # 左脚位姿
    right_foot_pose: Pose | None = None           # 右脚位姿
    foot_contact_left: bool = True                # 左脚接触状态
    foot_contact_right: bool = True               # 右脚接触状态
    support_phase: str = "double"                 # 支撑相: double|left_single|right_single
    com_position: tuple[float, float, float]      # 质心位置
    com_velocity: tuple[float, float, float]      # 质心速度

    # ===== 头部 =====
    head_pan_rad: float = 0.0                     # 头部水平转角
    head_tilt_rad: float = 0.0                    # 头部俯仰角
```

**为什么需要这些字段？**

人形机器人与轮式/四足机器人有本质区别：
- **ZMP (Zero Moment Point)**：必须时刻保持在支撑多边形内，否则会倾倒
- **支撑相 (Support Phase)**：决定了能否同时行走和执行操作（通常不能）
- **双臂独立力感知**：抓取和交接任务需要分别监控左右臂的力传感器
- **头部控制**：注视目标（gaze_at）技能依赖 head_pan/tilt

---

## 二、任务与规划模型 (`task.py` + `plan.py`)

### 2.1 TaskSpec — 高层任务描述

```python
class TaskSpec(BaseModel):
    task_id: str
    robot_id: str
    goal: str                     # 自然语言任务目标，如 "go to kitchen and pick up the cup"
    priority: int = 0             # 优先级（越小越高）
    deadline_sec: float | None    # 截止时间
    context: dict = {}            # 任务上下文
    metadata: dict = {}
```

### 2.2 SubTask — 原子子任务（DAG 节点）

```python
class SubTask(BaseModel):
    sub_task_id: str                        # 唯一标识符
    skill_type: str                         # 技能类型（对应 SkillType 枚举）
    parameters: dict = {}                    # 技能参数
    preconditions: list[str] = []           # 前置依赖的子任务 ID 列表
    expected_outcome: dict = {}              # 期望输出
    timeout_sec: float = 30.0               # 超时时间
    priority: int = 0
    retry_policy: str = "default"           # 重试策略: default | never | always_retry

    def dependencies_satisfied(self, completed_ids: set[str]) -> bool:
        """检查所有前置条件是否已满足"""
        return set(self.preconditions).issubset(completed_ids)
```

### 2.3 TaskGraph — DAG 任务图

```python
class TaskGraph(BaseModel):
    sub_tasks: list[SubTask] = []

    @property
    def task_ids(self) -> set[str]:
        return {t.sub_task_id for t in self.sub_tasks}

    def next_executable(self, completed_ids: set[str]) -> list[SubTask]:
        """返回所有前置条件已满足且未完成的子任务"""
        # 过滤出 preconditions ⊆ completed_ids 的子任务

    def is_complete(self, completed_ids: set[str]) -> bool:
        """所有子任务都完成？"""
        return self.task_ids.issubset(completed_ids)
```

**DAG 执行示例**：任务 "go to kitchen and pick up cup" 生成如下 DAG：

```
perceive_scene
       │
       ▼
navigate_to_kitchen ──────┐
       │                   │
       ▼                   ▼
locate_cup            safety_check
       │                   │
       └───────┬───────────┘
               ▼
          grasp_cup
```

`navigate_to_kitchen` 和 `safety_check` 可以在 `perceive_scene` 完成后**并行执行**。

### 2.4 ExecutionPlan — 完整执行计划

```python
class ExecutionPlan(BaseModel):
    plan_id: str
    task_goal: str
    sub_tasks: list[SubTask] = []
    estimated_duration_sec: float = 0.0
    created_at: float          # 创建时间戳
    status: str = "pending"
    completed_count: int = 0
    total_count: int = 0

    def progress(self) -> float:
        """完成百分比 0.0-1.0"""
```

---

## 三、感知模型 (`perception.py` + `sensor.py`)

### 3.1 传感器数据

```python
class CameraFrame(BaseModel):
    camera_id: str
    image_data: bytes | None    # RGB 图像数据
    depth_data: bytes | None    # 深度图
    resolution: tuple[int, int]
    intrinsics: dict

class LidarScan(BaseModel):
    points: list[list[float]]   # N×3 点云
    intensities: list[float]
    frame_id: str

class IMUReading(BaseModel):
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]
    orientation: tuple[float, float, float, float]

class ForceTorque(BaseModel):
    force_xyz: tuple[float, float, float]
    torque_xyz: tuple[float, float, float]

class ProprioceptiveState(BaseModel):
    joint_positions: dict[str, float]
    joint_velocities: dict[str, float]
    joint_torques: dict[str, float]
```

### 3.2 感知融合结果

```python
class DetectedObject(BaseModel):
    label: str
    confidence: float
    bbox_3d: dict             # 3D 边界框
    pose: Pose | None
    attributes: dict = {}     # 颜色、材质等属性

class SceneGraph(BaseModel):
    objects: list[DetectedObject]
    relationships: list[dict]  # 物体间关系，如 "cup ON table"

class PerceptionSnapshot(BaseModel):
    timestamp: float
    objects: list[DetectedObject]
    scene_graph: SceneGraph | None
    speech_text: str | None
    sound_events: list[str]
```

---

## 四、技能与情节模型 (`skill.py` + `episode.py`)

### 4.1 SkillDefinition — 技能定义

```python
class SkillParameter(BaseModel):
    name: str              # 参数名
    param_type: str        # 类型: string | int | float | bool
    description: str       # 中文说明
    required: bool = True
    default: Any = None

class SkillDefinition(BaseModel):
    name: str                                # 技能显示名
    skill_type: str                          # 对应 SkillType 枚举
    description: str                         # 功能描述
    parameters: list[SkillParameter] = []    # 参数列表
    timeout_sec: float = 30.0                # 默认超时
    tags: list[str] = []                     # 标签: [navigation, manipulation]
```

### 4.2 EpisodeRecord — 情节记录

```python
class EpisodeRecord(BaseModel):
    episode_id: str
    robot_id: str
    task_id: str | None
    task_type: str          # 任务类型
    goal: str               # 自然语言目标
    plan_id: str | None
    started_at: float
    ended_at: float | None
    outcome: str            # Outcome 枚举值
    summary: str
    initial_state: dict     # 初始状态快照
    final_state: dict       # 最终状态快照
    recovery_attempts: int = 0
    total_duration_sec: float = 0.0
```

---

## 五、记忆模型 (`memory.py`)

### LLMContext — 编译后的 LLM 上下文

```python
class LLMContext(BaseModel):
    robot_id: str
    task_goal: str
    current_state_summary: str           # 当前状态文本摘要
    current_phase: str
    relevant_episodes: list              # 相关历史情节
    relevant_semantic: list              # 语义检索结果
    relevant_knowledge: list             # 知识库查询结果
    spatial_context: dict                # 空间上下文
    safety_constraints: list[str]        # 安全约束列表

    def to_prompt_text(self) -> str:
        """将所有上下文编译为 LLM 提示词文本"""
```

这是上下文编译器的输出格式，被 TaskDecomposer 消费。

---

## 六、安全与 A2A 模型 (`safety.py` + `agent.py`)

### 6.1 安全模型

```python
class SafetyZone(BaseModel):
    zone_id: str
    geometry_type: str            # "sphere" | "box" | "cylinder"
    dimensions: dict
    severity: SafetySeverity

class ForceLimit(BaseModel):
    component: str                # 关节或传感器名
    limit_nm: float              # 力/力矩上限
    soft_limit_nm: float         # 软上限（预警值）

class SafetyViolation(BaseModel):
    event_id: str
    robot_id: str
    violation_type: str           # joint_limit | zmp_violation | force_limit | collision
    severity: SafetySeverity
    context: dict                 # 违反时的上下文数据
    timestamp: float
```

### 6.2 A2A 智能体模型

```python
class AgentCapability(BaseModel):
    skill_type: str
    proficiency: float = 1.0    # 熟练度 0.0-1.0
    max_payload_kg: float = 5.0

class AgentCard(BaseModel):
    agent_id: str
    display_name: str
    robot_model: str
    capabilities: list[AgentCapability]
    current_status: str          # idle | busy | error
    current_location: str
    version: str

class AgentMessage(BaseModel):
    from_agent: str
    to_agent: str
    message_type: str            # task_request | status_update | handover
    payload: dict
    timestamp: float
```
