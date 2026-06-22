# A2A 协议与通信层

支持多个人形机器人协作（"Swarm Robotics"）的 Agent-to-Agent 协议，以及底层通信基础设施。位于 `roboclaw/a2a/` 和 `roboclaw/comm/`。

---

## 一、A2A 架构

```
RoboClaw Agent A                      RoboClaw Agent B
┌──────────────┐                     ┌──────────────┐
│  A2A Server  │ ←── task request ── │  A2A Client  │
│  (接收委托)    │ ── status/result → │  (发送委托)    │
│              │                     │              │
│  Agent       │                     │  Agent       │
│  Registry    │ ←── discovery ────→ │  Registry    │
│              │                     │              │
│  Negotiation │ ←─ capability ───→ │  Negotiation │
│  Engine      │     matching        │  Engine      │
└──────────────┘                     └──────────────┘
```

遵循 **Google A2A 协议规范**的核心概念：
- Agent Card（`/.well-known/agent.json`）用于发现
- 任务提交/状态查询/结果流式推送
- 能力协商与负载均衡

---

## 二、A2A 服务端 (`server.py`)

```python
class A2AServer:
    def __init__(self, agent_id, agent_card_params):
        # Agent Card — 描述此 Agent 能力的标准化元数据
        self._agent_card = AgentCard(
            agent_id=agent_id,
            display_name=agent_card_params.get("display_name", agent_id),
            robot_model=agent_card_params.get("robot_model", "unknown"),
            capabilities=[...],          # 技能列表 + 熟练度
            current_status="idle",
            current_location="unknown",
            version="0.1.0",
        )
        self._active_tasks: dict[str, dict] = {}
        self._task_counter = 0

    async def accept_task(self, task_request: dict) -> dict:
        """接受一个来自同伴 Agent 的任务委托"""
        task_id = f"a2a_{self._agent_id}_{self._task_counter}"
        self._task_counter += 1
        task = {
            "task_id": task_id,
            "from_agent": task_request.get("from_agent"),
            "goal": task_request.get("goal"),
            "status": "accepted",
            "created_at": time(),
        }
        self._active_tasks[task_id] = task
        return task

    async def get_task_status(self, task_id: str) -> dict:
        """查询委托任务的状态"""
        ...

    async def cancel_task(self, task_id: str) -> dict:
        """取消委托任务"""
        ...
```

### Agent Card 结构（A2A 规范）

```json
{
  "agent_id": "h1_unitree_01",
  "display_name": "H1 Kitchen Assistant",
  "robot_model": "h1_unitree",
  "capabilities": [
    {"skill_type": "whole_body_grasp", "proficiency": 0.9, "max_payload_kg": 5.0},
    {"skill_type": "navigate_to", "proficiency": 0.95},
    {"skill_type": "handover", "proficiency": 0.85}
  ],
  "current_status": "idle",
  "current_location": "kitchen",
  "version": "0.1.0"
}
```

---

## 三、A2A 客户端 (`client.py`)

```python
class A2AClient:
    def __init__(self):
        self._known_agents: dict[str, AgentCard] = {}
        self._stub_mode = True  # Phase 1 stub

    async def discover_agent(self, agent_url: str) -> AgentCard | None:
        """通过 GET /.well-known/agent.json 发现同伴"""
        # stub: 直接返回模拟 AgentCard

    async def submit_task(self, agent_id: str, task_spec: dict) -> dict:
        """向同伴提交任务委托"""
        # stub: 模拟提交，总是成功

    async def get_task_result(self, agent_id: str, task_id: str) -> dict:
        """获取委托任务的执行结果"""
        # stub: 返回模拟成功结果
```

---

## 四、Agent 注册表 (`registry.py`)

```python
class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentCard] = {}

    async def register(self, agent_card: AgentCard):
        self._agents[agent_card.agent_id] = agent_card

    async def unregister(self, agent_id: str):
        self._agents.pop(agent_id, None)

    async def find_by_capability(self, required_skill: str, min_proficiency=0.5) -> list[AgentCard]:
        """查找具备特定能力且熟练度达标的同伴"""
        results = []
        for agent in self._agents.values():
            if agent.current_status != "idle":
                continue
            for cap in agent.capabilities:
                if cap.skill_type == required_skill and cap.proficiency >= min_proficiency:
                    results.append(agent)
                    break
        return results

    async def list_available(self) -> list[AgentCard]:
        return [a for a in self._agents.values() if a.current_status == "idle"]
```

---

## 五、协商引擎 (`negotiation.py`)

```python
class NegotiationEngine:
    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def find_best_agent(self, task_spec: dict) -> AgentCard | None:
        """为给定任务找到最合适的 Agent（能力匹配 + 分数排名）"""
        required_skill = task_spec.get("skill_type", "")
        candidates = await self._registry.find_by_capability(required_skill)

        if not candidates:
            return None

        # 按匹配分数降序排列
        scored = []
        for candidate in candidates:
            score = self._score_match(task_spec, candidate)
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def _score_match(self, task_spec, agent_card) -> float:
        """计算 Agent 与任务需求的匹配分数"""
        score = 0.0
        for cap in agent_card.capabilities:
            if cap.skill_type == task_spec.get("skill_type", ""):
                score += cap.proficiency * 2.0       # 技能熟练度（权重×2）
            else:
                score += cap.proficiency * 0.2        # 其他技能（低权重）
        # 负载惩罚（busy 状态 Agent 降分）
        if agent_card.current_status == "busy":
            score *= 0.5
        return score
```

### 协商流程

```
1. Agent A 收到任务 "pick up cup in kitchen"
2. Agent A 评估自身能力（不在厨房位置）
3. 调度 negotiate.find_best_agent({"skill_type": "whole_body_grasp", "location": "kitchen"})
4. 协商引擎 → Registry.find_by_capability("whole_body_grasp")
5. 找到 Agent B (在厨房，空闲，抓取熟练度 0.9)
6. 返回 Agent B
7. Agent A → A2AClient.submit_task("Agent B", task)
8. Agent A 的状态图上 should_delegate 标记，执行 delegate_a2a_node
```

---

## 六、通信层 (`comm/`)

### 6.1 ROS 2 桥接 (`ros2_bridge.py`)

**Phase 1 Stub** — 文档化了预期的 ROS 2 集成方式：

```python
class ROS2Bridge:
    """ROS 2 与 RoboClaw 框架之间的桥接层"""

    # 预期发布的 Topic:
    # /joint_states (sensor_msgs/JointState)
    # /imu/data (sensor_msgs/Imu)
    # /camera/rgb (sensor_msgs/Image)
    # /camera/depth (sensor_msgs/Image)
    # /force_torque/left (geometry_msgs/WrenchStamped)
    # /force_torque/right (geometry_msgs/WrenchStamped)

    # 预期订阅的 Topic:
    # /joint_commands (trajectory_msgs/JointTrajectory)

    # 预期使用的 Action:
    # /walk_to_pose (Action: Pose → Result: success)
    # /grasp_object (Action: ObjectID + GraspType → Result: success/detail)
    # /place_object (Action: TargetPose → Result: success/detail)
```

### 6.2 RabbitMQ 客户端 (`rabbitmq.py`)

**Phase 1 Stub** — 模拟消息队列操作：

```python
class RabbitMQClient:
    def __init__(self, config: RabbitMQConfig):
        self._config = config
        self._stub_mode = True

    async def connect(self):
        """连接到 RabbitMQ (stub: 总是成功)"""

    async def publish(self, exchange: str, routing_key: str, message: dict):
        """发布消息到 Exchange"""
        logger.info(f"[STUB] Publish to {exchange}/{routing_key}: {message}")

    async def subscribe(self, exchange: str, routing_key: str, callback):
        """订阅消息"""
        logger.info(f"[STUB] Subscribed to {exchange}/{routing_key}")

    async def close(self):
        """关闭连接"""
```

**预期的事件 Exchange**：

| Exchange | Routing Key | 用途 |
|----------|------------|------|
| `roboclaw.sensors` | `sensor.camera.{id}` | 相机帧事件 |
| `roboclaw.sensors` | `sensor.imu.{robot_id}` | IMU 数据事件 |
| `roboclaw.tasks` | `task.created` / `task.completed` | 任务生命周期事件 |
| `roboclaw.episodes` | `episode.created` / `episode.completed` | 情节记录事件 |
| `roboclaw.a2a` | `a2a.task_request.{agent_id}` | A2A 任务委托事件 |
| `roboclaw.safety` | `safety.violation.{severity}` | 安全违规事件 |

### 6.3 MQTT 客户端 (`mqtt_client.py`)

**Phase 1 Stub** — 为轻量级传感器遥测预留（IoT 场景）。

---

## 七、多 Agent 协作示例

`examples/demo_a2a_swarm.py` 演示了两台 H1 人形机器人协作：

```
Agent A (h1_kitchen_01) — 位于客厅
   │
   │ 任务: "go to the kitchen and pick up the cup"
   │
   ├─ 自身能力评估: 不在厨房，需要走到厨房再执行
   │
   ├─ A2A 协商: find_best_agent("whole_body_grasp", "kitchen")
   │    └─ 发现 Agent B (h1_kitchen_02) — 在厨房，空闲，熟练度 0.9
   │
   ├─ A2A 委托: submit_task(Agent B, "pick up cup in kitchen")
   │
   └─ Agent B 执行委托任务，完成后返回结果给 Agent A
```

这种模式允许：
- **负载分散**：繁忙的 Agent 将子任务委托给空闲的同伴
- **空间优化**：优先委派离目标位置最近的 Agent
- **能力互补**：Agent 可以请求具备特定技能或更大负载能力的同伴
