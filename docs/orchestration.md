# 编排引擎 (LangGraph StateGraph)

编排引擎是 RoboClaw 的"中枢神经系统"，基于 LangGraph 的 StateGraph 构建。位于 `roboclaw/orchestration/`。

---

## 一、Agent 图状态 (`state_schema.py`)

`AgentGraphState` 是流经整个 LangGraph 状态图的**唯一真相源**，被 PostgreSQL/LangGraph checkpointer 持久化，支持故障恢复。

```python
class AgentGraphState(BaseModel):
    # ========== 身份标识 ==========
    robot_id: str = ""
    session_id: str = Field(default_factory=_new_session_id)  # 自动生成

    # ========== 任务与规划 ==========
    task_spec: dict | None = None               # 高层任务描述
    execution_plan: dict | None = None           # LLM 生成的执行计划
    current_subtask: dict | None = None          # 当前正在执行的子任务
    completed_subtask_ids: list[str] = []        # 已完成的子任务 ID
    task_history: list[dict] = []                # 历史任务（同一 session）

    # ========== 感知 ==========
    perception: PerceptionData | None = None     # 最新感知快照
    latest_state_summary: str = ""               # 状态文本摘要

    # ========== 世界状态 ==========
    robot_pose: dict | None = None
    world_objects: list[dict] = []
    safety_status: dict = {}

    # ========== 记忆指针 ==========
    episode_id: str | None = None               # 当前 Episode ID
    relevant_memories: list[dict] = []           # 检索到的相关记忆

    # ========== 对话/HRI ==========
    messages: Annotated[list, add_messages] = []  # LangGraph add_messages reducer

    # ========== 控制流 ==========
    loop_count: int = 0                          # 循环计数
    phase: str = "idle"                          # 当前 AgentPhase
    error: dict | None = None                    # 当前错误信息
    recovery_attempts: int = 0                   # 已尝试恢复次数
    max_recovery_attempts: int = 3               # 最大恢复尝试
    should_continue: bool = True                 # 是否继续主循环
    should_delegate: bool = False                # 是否委托给同伴
    require_human_help: bool = False             # 是否需要人类帮助

    # ========== A2A 多智能体 ==========
    peer_agents: list[dict] = []                 # 已知的同伴 Agent
    pending_delegations: list[dict] = []         # 待处理的委托

    model_config = ConfigDict(arbitrary_types_allowed=True)
```

### 关键点

1. **`messages` 字段使用 LangGraph 的 `add_messages` reducer**：这意味着多个节点对 messages 的更新是**追加**而非覆盖，适合对话式交互
2. **`loop_count` 用于防止无限循环**：在每个 PERCEIVE 节点递增
3. **`arbitrary_types_allowed=True`**：允许 Pydantic 模型包含非标准类型（如 `add_messages`）

---

## 二、节点函数 (`nodes.py`)

LangGraph 中的每个节点都是纯函数：`(state) -> dict of state updates`。节点本身很薄，负责将任务分派给各自的子模块。

### 2.1 perceive_node — 感知节点

```python
async def perceive_node(state: AgentGraphState) -> dict[str, Any]:
    """摄取多模态传感器数据，生成统一的感知快照"""
    # 生产环境：调用 PerceptionFusion.process()
    # Phase 1：从 state.world_objects 模拟感知
    perception = PerceptionData(
        timestamp=time(),
        detected_objects=state.world_objects,
    )
    return {
        "perception": perception,
        "phase": AgentPhase.PERCEIVING,
        "loop_count": state.loop_count + 1,  # 递增循环计数
    }
```

### 2.2 reflect_node — 反思节点

```python
async def reflect_node(state: AgentGraphState) -> dict[str, Any]:
    """更新工作记忆、检查安全约束、记录关键变化"""
    # 生成状态摘要文本
    objects_seen = len(state.perception.detected_objects)
    summary = f"Objects detected: {objects_seen}. Phase: {state.phase}."

    # 安全检查（检查 e-stop 状态）
    safety_ok = not state.safety_status.get("estop", False)

    return {
        "latest_state_summary": summary,
        "phase": AgentPhase.REFLECTING,
        "should_continue": safety_ok,  # e-stop 激活时停止循环
    }
```

### 2.3 plan_node — 规划节点

```python
async def plan_node(state: AgentGraphState) -> dict[str, Any]:
    """将高层任务分解为可执行的 SubTask DAG"""
    # 生产环境：调用 TaskDecomposer + ContextCompiler
    # Phase 1：生成固定三步计划
    plan = {
        "plan_id": f"plan_{state.session_id}",
        "task_goal": goal,
        "sub_tasks": [
            {"sub_task_id": "perceive_scene", "skill_type": "gaze_at", ...},
            {"sub_task_id": "navigate_to_target", "skill_type": "navigate_to",
             "preconditions": ["perceive_scene"], ...},
            {"sub_task_id": "grasp_object", "skill_type": "whole_body_grasp",
             "preconditions": ["navigate_to_target"], ...},
        ],
    }
    return {"execution_plan": plan, "phase": AgentPhase.PLANNING}
```

### 2.4 act_node — 执行节点

```python
async def act_node(state: AgentGraphState) -> dict[str, Any]:
    """执行下一个前置条件已满足的子任务"""
    # 1. 收集已完成的子任务 ID
    completed = set(state.completed_subtask_ids)
    # 2. 找到下一个可执行的子任务（所有 preconditions 都在 completed 中）
    for st in plan["sub_tasks"]:
        if st["sub_task_id"] not in completed \
           and set(st["preconditions"]).issubset(completed):
            next_subtask = st
            break
    return {"current_subtask": next_subtask, "phase": AgentPhase.ACTING}
```

### 2.5 feedback_node — 反馈节点

```python
async def feedback_node(state: AgentGraphState) -> dict[str, Any]:
    """评估子任务执行结果，决定下一步：继续/恢复/委托/HRI/完成"""
    current = state.current_subtask

    # 没有待执行子任务且无错误 → 完成
    if current is None and state.error is None:
        return {"phase": AgentPhase.EVALUATING, "should_continue": False}

    # 标记当前子任务为已完成
    if current:
        completed = list(state.completed_subtask_ids) + [current["sub_task_id"]]
        return {"completed_subtask_ids": completed, "current_subtask": None,
                "phase": AgentPhase.EVALUATING}
```

### 2.6 recover_node — 恢复节点

```python
async def recover_node(state: AgentGraphState) -> dict[str, Any]:
    """尝试恢复（责任链模式）"""
    recovery_attempts = state.recovery_attempts + 1

    # 达到最大尝试次数 → 请求人类帮助
    if recovery_attempts >= state.max_recovery_attempts:
        return {"recovery_attempts": recovery_attempts,
                "require_human_help": True}

    # 模拟恢复逻辑：对已知错误类型自动恢复
    if error_type in ("kinematic", "dynamic", "perception"):
        return {"recovery_attempts": recovery_attempts, "error": None}  # 清除错误

    # 未知错误 → 升级
    return {"recovery_attempts": recovery_attempts, "require_human_help": True}
```

### 2.7 delegate_a2a_node / hri_node

委托和 HRI 节点在 Phase 1 中为 stub 实现，仅设置 phase 即可。生产环境将分别调用 A2A 客户端和 WebSocket 消息推送。

---

## 三、条件路由 (`edges.py`)

三条条件边函数决定了 Agent 循环的控制流。

### route_after_feedback — 反馈后的路由

```
反馈节点完成 → 检查状态 →
    ├── should_continue == False → "complete" (→ END)
    ├── 有 error 且未超恢复次数 → "recover" (→ recover_node)
    ├── 有 error 且 require_human_help → "hri" (→ handle_hri)
    ├── should_delegate 且有 peer → "delegate" (→ delegate_a2a)
    └── 默认 → "continue" (→ perceive_node，继续主循环)
```

```python
def route_after_feedback(state: AgentGraphState) -> str:
    if not state.should_continue: return "complete"
    if state.error:
        if state.recovery_attempts < state.max_recovery_attempts:
            return "recover"
        if state.require_human_help: return "hri"
        return "complete"  # 不可恢复，终止
    if state.should_delegate and state.peer_agents: return "delegate"
    return "continue"
```

### route_after_recovery — 恢复后的路由

```
恢复完成 → 检查结果 →
    ├── error 已清除 → "retry" (→ act_node，重试失败的子任务)
    ├── require_human_help → "human_help" (→ handle_hri)
    ├── error 类型是 planning/kinematic → "replan" (→ plan_node)
    └── 否则 → "abort" (→ END)
```

### route_after_hri — HRI 后的路由

```
HRI 完成 → 检查结果 →
    ├── error 已清除（操作员解决了问题） → "continue" (→ perceive)
    └── 仍有错误 → "abort" (→ END)
```

---

## 四、图构建 (`graph.py`)

### build_agent_graph() — 核心工厂函数

```python
def build_agent_graph(checkpointer=None) -> StateGraph:
    graph = StateGraph(AgentGraphState)

    # ===== 注册 8 个节点 =====
    graph.add_node("perceive", perceive_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("recover", recover_node)
    graph.add_node("delegate_a2a", delegate_a2a_node)
    graph.add_node("handle_hri", hri_node)

    graph.set_entry_point("perceive")  # 入口节点

    # ===== 主循环的固定边 =====
    graph.add_edge("perceive", "reflect")
    graph.add_edge("reflect", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "feedback")

    # ===== 条件边（五路分支） =====
    graph.add_conditional_edges("feedback", route_after_feedback, {
        "continue": "perceive", "recover": "recover",
        "delegate": "delegate_a2a", "hri": "handle_hri", "complete": END,
    })
    graph.add_conditional_edges("recover", route_after_recovery, {
        "replan": "plan", "retry": "act",
        "abort": END, "human_help": "handle_hri",
    })

    # ===== 返回边 =====
    graph.add_edge("handle_hri", "perceive")       # HRI 完成后重新感知
    graph.add_edge("delegate_a2a", "feedback")     # 委托后进入反馈

    return graph.compile(checkpointer=checkpointer)
```

### create_agent_runner() — 快捷生成器

```python
async def create_agent_runner(robot_id, checkpointer=None):
    graph = build_agent_graph(checkpointer)
    config = {
        "configurable": {
            "thread_id": f"roboclaw-{robot_id}",     # 每个机器人独立线程
            "checkpoint_ns": robot_id,
        }
    }
    return graph, config
```

---

## 五、检查点持久化 (`checkpointer.py`)

```python
# 生产：PostgreSQL 持久化（支持进程重启后恢复）
async def create_postgres_checkpointer(conn_string: str) -> PostgresSaver | None:
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string(conn_string)
    await checkpointer.setup()
    return checkpointer

# 开发/测试：内存检查点
def create_memory_checkpointer() -> MemorySaver:
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
```

---

## 六、子图 (`subgraphs.py`)

可重用的复合行为子图，可嵌入主图或互相组合。

### 抓取子图（Grasp Subgraph）

```
grasp_approach → grasp_execute → grasp_verify → (retry|complete)
       ↑                                            │
       └──────────── (retry) ───────────────────────┘
```

三个节点：
- `grasp_approach`：接近目标物体
- `grasp_execute`：执行抓取动作
- `grasp_verify`：检查抓取稳定性（失败则 retry → approach）

### 步行子图（Walk Subgraph）

```
walk_plan → walk_execute → walk_balance → (retry|complete)
    ↑                                         │
    └─────────── (retry) ─────────────────────┘
```

三个节点：
- `walk_plan`：规划脚步序列
- `walk_execute`：执行一步
- `walk_balance`：检查平衡（ZMP 是否在支撑多边形内）
