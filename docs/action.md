# 执行层 (Action Layer)

执行层管理技能的定义、调度和（模拟）执行。位于 `roboclaw/action/`。

---

## 一、基础抽象 (`base.py`)

```python
class ActionResultStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"  # 被取消或中断

class ActionResult(BaseModel):
    sub_task_id: str
    status: ActionResultStatus
    duration_sec: float = 0.0
    actual_outcome: dict = {}
    error: dict | None = None

class AbstractActionExecutor(ABC):
    @abstractmethod
    async def execute(self, sub_task: dict, world_state: dict) -> ActionResult: ...
    async def cancel(self) -> None: ...
    async def emergency_stop(self) -> None: ...
```

所有具体执行器（LocomotionExecutor、ManipulationExecutor等）都继承此基类，保证统一的接口。

---

## 二、技能库 (`skill_library.py`)

技能库是执行层的**总调度入口**。它维护技能定义 → 执行器实现的映射关系。

### 9 种内置人形机器人技能

| 技能名 | SkillType | 关键参数 | 超时 | 分类 |
|--------|-----------|---------|------|------|
| **Navigate to Room** | navigate_to | room, speed | 120s | 导航/运动 |
| **Walk Steps** | walk_steps | num_steps, direction, step_length | 30s | 运动 |
| **Whole-Body Grasp** | whole_body_grasp | object, arm, grasp_type | 30s | 操作/抓取 |
| **Place Object** | place_object | object, target_xyz | 20s | 操作 |
| **Handover** | handover | object, receiver | 30s | 操作/交互 |
| **Speak** | speak | text, volume | 10s | 交互/通信 |
| **Gaze At** | gaze_at | target | 5s | 感知 |
| **Open Door** | open_door | door, arm | 45s | 操作/导航 |
| **Climb Stairs** | climb_stairs | num_steps, direction | 60s | 运动 |

### 技能调度

```python
async def execute_skill(skill_type, parameters, world_state=None) -> ActionResult:
    # 1. 查找技能定义
    definition = self._skills.get(skill_type)
    if definition is None:
        return ActionResult(status=FAILURE, error={"type": "unknown_skill"})

    # 2. 查找执行器
    executor = self._executors.get(skill_type)
    if executor is None:
        # Phase 1：无真实执行器时模拟执行
        logger.info(f"Simulating execution of skill: {skill_type}")
        return ActionResult(status=SUCCESS, actual_outcome={"simulated": True})

    # 3. 真实执行
    try:
        result = await executor.execute(task_dict, world_state)
        return result
    except Exception as e:
        return ActionResult(status=FAILURE, error={"type": "execution_error", "detail": str(e)})
```

### LangChain 工具绑定

```python
def as_langchain_tools(self) -> list:
    """将技能导出为 LangChain StructuredTool，供 LLM 调用"""
    for skill_type, definition in self._skills.items():
        tool = StructuredTool(
            name=skill_type,
            description=definition.description,
            coroutine=_execute,  # 包装了 execute_skill 的异步函数
        )
```

这允许 LLM Agent 直接通过 LangChain 工具调用接口执行机器人技能。

---

## 三、运动执行器 (`locomotion.py`)

负责双足行走和平衡控制，Phase 1 为模拟实现。

```python
class LocomotionExecutor(AbstractActionExecutor):
    async def execute(self, sub_task, world_state) -> ActionResult:
        # 按技能类型分发
        if skill_type == "navigate_to":   return await self._navigate_to(params)
        if skill_type == "walk_steps":    return await self._walk_steps(params)
        if skill_type == "climb_stairs":  return await self._climb_stairs(params)

    async def _navigate_to(self, params) -> ActionResult:
        """模拟走到某个房间"""
        room = params.get("room", "unknown")
        # 模拟 5 秒行走（每 0.1s 检查一次取消标志）
        for i in range(int(5.0)):
            if self._cancelled: break
            await asyncio.sleep(0.1)
        return ActionResult(status=SUCCESS, actual_outcome={"at_location": room})

    async def _walk_steps(self, params) -> ActionResult:
        """模拟走指定步数"""
        num_steps = params.get("num_steps", 1)
        duration = num_steps * 0.6  # 每步 0.6 秒
        return ActionResult(status=SUCCESS, actual_outcome={"steps_completed": num_steps})

    async def cancel(self) -> None:
        """优雅停止：完成当前步后停止"""
        self._cancelled = True

    async def emergency_stop(self) -> None:
        """紧急停止：立即停止"""
        self._cancelled = True
```

**关键设计**：
- `cancel()` vs `emergency_stop()`：生产环境中 cancel 完成当前步态周期后安全停止，estop 立即切断动力
- `_cancelled` 标志在 `asyncio.sleep` 循环中每 0.1s 检查一次
- 生产环境将通过 ROS 2 JointTrajectory Action 控制真实关节

---

## 四、操作执行器 (`manipulation.py`)

负责双臂操作（抓取、放置、交接、开门），Phase 1 为模拟实现。

```python
class ManipulationExecutor(AbstractActionExecutor):
    async def execute(self, sub_task, world_state) -> ActionResult:
        # 按技能类型分发
        if skill_type == "whole_body_grasp": return await self._grasp(params)
        if skill_type == "place_object":     return await self._place(params)
        if skill_type == "handover":         return await self._handover(params)
        if skill_type == "open_door":        return await self._open_door(params)

    async def _grasp(self, params) -> ActionResult:
        """模拟全身协调抓取"""
        obj = params.get("object", "unknown")
        arm = params.get("arm", "right")
        grasp_type = params.get("grasp_type", "top_down_encompassing")
        # 模拟 3s 抓取过程
        await asyncio.sleep(1.0)
        return ActionResult(
            status=SUCCESS,
            actual_outcome={"grasped": True, "object": obj, "arm": arm}
        )

    async def _handover(self, params) -> ActionResult:
        """模拟物体交接"""
        obj = params.get("object", "unknown")
        receiver = params.get("receiver", "human")
        await asyncio.sleep(1.0)
        return ActionResult(
            status=SUCCESS,
            actual_outcome={"handed_over": True, "object": obj, "receiver": receiver}
        )
```

**生产环境对接**：
- 抓取：RGB-D 相机 + GraspNet → IK 求解 → 关节轨迹生成
- 力控：FT 传感器反馈 → 阻抗控制器
- ROS 2 Action：`/grasp_action`, `/place_action` 等

---

## 五、其他执行器

### head_neck.py — 头部控制

```python
class HeadNeckExecutor(AbstractActionExecutor):
    # 技能：gaze_at(target) — 控制 pan/tilt 使相机对准目标
    # 生产环境：通过 ROS 2 JointPositionController 控制头部关节
```

### speech.py — 语音输出

```python
class SpeechExecutor(AbstractActionExecutor):
    # 技能：speak(text, volume) — 通过 TTS 输出语音
    # 生产环境：集成 Edge TTS / ElevenLabs API
```

### navigation.py — 导航

```python
class NavigationExecutor(AbstractActionExecutor):
    # 技能：navigate_to(room), walk_steps(n), climb_stairs(n)
    # 通常委托给 LocomotionExecutor，需要时添加路径跟踪逻辑
```
