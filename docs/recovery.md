# 故障恢复系统 (Recovery Subsystem)

机器人在真实环境中执行任务时不可避免会遇到故障。RoboClaw 的恢复系统采用**责任链模式**，从自动化修复逐级升级到人工介入。位于 `roboclaw/recovery/`。

---

## 一、恢复结果与策略基类 (`base.py`)

```python
class RecoveryResult(BaseModel):
    success: bool = False                       # 恢复是否成功
    strategy_used: str = ""                     # 使用的策略名
    modified_subtask: dict | None = None         # 调整后的子任务参数（重试用）
    recovery_event: dict | None = None           # 恢复事件记录
    human_message: str | None = None             # 给操作员的消息

class AbstractRecoveryStrategy(ABC):
    @abstractmethod
    def can_handle(self, error: dict, state: Any) -> bool:
        """检查此策略是否能处理该错误"""

    @abstractmethod
    async def handle(self, error: dict, state: Any) -> RecoveryResult:
        """尝试恢复"""
```

---

## 二、恢复责任链 (`recovery_chain.py`)

### 链结构

```
SelfDiagnose → RePlan → RetrySkill → SafeStop → HumanHelp
    (优先)                                      (最后防线)
```

策略按注册顺序执行。第一个 `can_handle()=True` 的策略被调用。如果它失败，链继续到下一个策略。

### 核心逻辑

```python
class RecoveryChain:
    def __init__(self):
        self._strategies: list[AbstractRecoveryStrategy] = []

    async def attempt_recovery(self, error: dict, state: Any) -> RecoveryResult:
        logger.info(f"Starting recovery chain for error: {error.get('type')}")

        for i, strategy in enumerate(self._strategies):
            # 跳过不能处理此错误的策略
            if not strategy.can_handle(error, state):
                logger.debug(f"Strategy '{strategy.__class__.__name__}' cannot handle, skipping")
                continue

            # 尝试执行
            logger.info(f"Attempting '{strategy.__class__.__name__}' ({i+1}/{len(self._strategies)})")
            try:
                result = await strategy.handle(error, state)
                if result.success:
                    logger.info(f"Recovery successful via '{result.strategy_used}'")
                    return result
                logger.warning(f"Strategy '{result.strategy_used}' failed, trying next")
            except Exception as e:
                logger.error(f"Strategy raised exception: {e}")

        # 所有策略都失败了
        logger.error("All recovery strategies exhausted")
        return RecoveryResult(
            success=False,
            strategy_used="exhausted",
            human_message="All recovery strategies exhausted. Manual intervention required.",
        )
```

### 关键设计

1. **HumanHelp 永远是链中最后一个**：它的 `can_handle()` 始终返回 `True`，确保链永远不会进入死胡同
2. **每个策略成功时立即返回**：不继续尝试后面的策略
3. **策略异常被捕获**：一个策略抛异常不会中断整个恢复链
4. **human_message 字段**：最终结果携带人类可读的错误描述

---

## 三、五种恢复策略

### 3.1 SelfDiagnose（自我诊断）

```python
class SelfDiagnoseStrategy(AbstractRecoveryStrategy):
    def can_handle(self, error, state) -> bool:
        # 处理通信错误和感知错误
        return error.get("type") in ("communication", "perception")

    async def handle(self, error, state) -> RecoveryResult:
        # 尝试重新连接传感器/通信通道
        # 测试 connectivity → 重置连接 → 重试数据流
        return RecoveryResult(success=True, strategy_used="self_diagnose")
```

**适用场景**：传感器超时、ROS 节点失联、消息队列断连等。

### 3.2 RePlan（重新规划）

```python
class RePlanStrategy(AbstractRecoveryStrategy):
    def can_handle(self, error, state) -> bool:
        # 处理运动学、动力学、规划、感知错误
        return error.get("type") in ("kinematic", "dynamic", "planning", "perception")

    async def handle(self, error, state) -> RecoveryResult:
        # 查询语义记忆中的类似故障模式
        # 基于当前世界状态重新分解任务
        return RecoveryResult(success=True, strategy_used="replan")
```

**适用场景**：目标物体被移动、IK 无解、路径被阻挡、计划超时。

### 3.3 RetrySkill（重试技能）

```python
class RetrySkillStrategy(AbstractRecoveryStrategy):
    def can_handle(self, error, state) -> bool:
        # 几乎总是可以尝试重试
        return True

    async def handle(self, error, state) -> RecoveryResult:
        # 调整参数后重试：
        # - 运动学错误 → 换手 (right → left)
        # - 感知错误 → 扩大搜索区域
        # - 超时错误 → 降低速度
        modified = dict(state.current_subtask)
        if error.get("type") == "kinematic":
            modified["parameters"]["arm"] = "left"  # 换手
        return RecoveryResult(
            success=True, strategy_used="retry_skill",
            modified_subtask=modified
        )
```

**适用场景**：抓取失败（换手）、导航失败（降速）、感知失败（扩大搜索范围）。

### 3.4 SafeStop（安全停止）

```python
class SafeStopStrategy(AbstractRecoveryStrategy):
    def can_handle(self, error, state) -> bool:
        # 处理安全错误和硬件错误
        return error.get("type") in ("safety", "hardware")

    async def handle(self, error, state) -> RecoveryResult:
        # 1. 立即停止所有运动
        # 2. 回到安全姿态（safe_pose）
        # 3. 启动诊断序列
        return RecoveryResult(success=True, strategy_used="safe_stop")
```

**适用场景**：碰撞检测、ZMP 越界、力/力矩超限、电机过热。

### 3.5 HumanHelp（请求人类帮助）

```python
class HumanHelpStrategy(AbstractRecoveryStrategy):
    def can_handle(self, error, state) -> bool:
        # 总是返回 True — 这是最后的安全网
        return True

    async def handle(self, error, state) -> RecoveryResult:
        # 组装详细的操作员消息
        human_message = (
            f"Robot {state.robot_id} requires assistance.\n"
            f"Task: {state.task_spec.get('goal', '')}\n"
            f"Error Type: {error.get('type', 'unknown')}\n"
            f"Error Detail: {error.get('detail', 'No additional detail')}\n"
            f"Failed Sub-Task: {state.current_subtask}\n"
            f"Recovery Attempts: {state.recovery_attempts}\n\n"
            f"Please review and resolve via the operator console."
        )
        return RecoveryResult(
            success=True,   # 升级到人类是成功的处理（从链的角度看）
            strategy_used="human_help",
            human_message=human_message,
        )
```

**关键设计决策**：`HumanHelpStrategy.handle()` 返回 `success=True`。虽然从机器人的角度看问题未自动解决，但从恢复链的角度看，成功地升级到了正确的一方（人类操作员）。如果返回 `success=False`，恢复链会继续并进入 "exhausted" 状态，这是无意义的行为。

---

## 四、重试策略 (`retry_policy.py`)

可配置的指数退避重试机制：

```python
class RetryPolicy:
    def __init__(self, max_retries=3, base_delay_s=0.5, max_delay_s=30.0,
                 backoff_multiplier=2.0, jitter=True):
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter  # 抖动避免雷鸣效应
        self._skill_overrides: dict[str, dict] = {}  # 按技能覆盖

    def get_delay(self, attempt: int) -> float:
        """delay = min(base * multiplier^(attempt-1), max_delay) + jitter"""
        delay = self.base_delay_s * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_s)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    def set_skill_policy(self, skill_name, max_retries, base_delay_s):
        """为特定技能设置独立的重试策略"""
        self._skill_overrides[skill_name] = {
            "max_retries": max_retries, "base_delay_s": base_delay_s
        }
```

**示例**：
- `get_delay(1)` → 0.5s (base)
- `get_delay(2)` → 1.0s (0.5 × 2)
- `get_delay(3)` → 2.0s (0.5 × 4)
- `get_delay(7)` → 30.0s (capped at max)

**技能特殊覆盖**：
```python
rp.set_skill_policy("whole_body_grasp", max_retries=5, base_delay_s=2.0)
# 抓取最多重试 5 次，从 2s 起步
```

---

## 五、安全监控器 (`safety_monitor.py`)

作为**后台协程**运行，以高频（默认 100Hz）检查安全约束。

### 三项核心检查

```python
class SafetyMonitor:
    async def _check_safety(self, state_provider):
        await self._check_joint_limits(state)    # 关节限位
        await self._check_zmp(state)              # ZMP 支撑多边形
        await self._check_forces(state)            # 力/力矩阈值
```

### 关节限位检查

```python
async def _check_joint_limits(self, state):
    joint_states = state.get("joint_positions", {})
    for joint_name, position in joint_states.items():
        if abs(position) > 3.0:  # 硬编码安全阈值
            self._add_violation("joint_limit", SafetySeverity.CRITICAL, {
                "joint": joint_name, "position": position
            })
```

### ZMP 在多边形内的判断（射线法）

```python
@staticmethod
def _point_in_polygon(point, polygon) -> bool:
    """Ray-casting algorithm — 经典算法判断点是否在凸/凹多边形内"""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # 射线从左到右穿过边的条件
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
```

### 安全状态查询

```python
@property
def is_safe(self) -> bool:
    """检查最近 10 个安全违规中是否有 CRITICAL 或 E_STOP 级别"""
    return not self._estop_active and not any(
        v.severity in (SafetySeverity.CRITICAL, SafetySeverity.E_STOP)
        for v in self._violations[-10:]
    )
```

### 急停逻辑

当检测到 `SafetySeverity.E_STOP` 级别的违规时：
1. 设置 `_estop_active = True`
2. `is_safe` 属性立即返回 `False`
3. 编排引擎的 `reflect_node` 检测到 `should_continue = False`，终止主循环
