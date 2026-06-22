# 规划层 (Planning Layer)

规划层负责将自然语言任务分解为可执行的 SubTask DAG。位于 `roboclaw/planning/`。

---

## 一、任务分解器 (`task_decomposer.py`)

框架中最核心的规划组件，将自然语言任务转换为结构化的执行计划。

### 架构

```
输入：TaskSpec (goal: "go to kitchen and pick up the cup")
  │
  ├── LLM 可用？ → _llm_plan()
  │     │
  │     ├── 编译 ContextCompiler 输出为提示词
  │     ├── LangChain ChatModel.ainvoke()
  │     ├── 从 LLM 回复中提取 JSON
  │     ├── 构造 ExecutionPlan
  │     └── PlanValidator.validate()
  │
  └── LLM 不可用？ → _template_plan()
        │
        ├── "pick up" → _pick_and_place_plan()
        ├── "go to"   → _navigation_plan()
        ├── "find"    → _search_plan()
        └── 其他       → _generic_plan()
```

### System Prompt 设计

LLM 规划的系统提示词定义了：

1. **10 种可用技能**：navigate_to, walk_to, locate_object, whole_body_grasp, place_object, handover, speak, gaze_at, open_door, press_button, wait_for
2. **输出 JSON 结构**：包含 plan_id, task_goal, 以及 sub_tasks 数组
3. **7 条规划规则**：
   - 每个 sub_task_id 唯一且描述性强
   - 前置条件形成有效 DAG（无环）
   - 总是从感知开始
   - 主要动作之间插入安全检查
   - 考虑双足平衡（避免同时行走和操作）
   - 明确指定使用的手（左/右）
   - 根据动作复杂度估算合理超时

### LLM 响应解析

```python
async def _llm_plan(self, goal: str, context: Any) -> ExecutionPlan:
    # 1. 组装提示词
    messages = [SystemMessage(self.SYSTEM_PROMPT), HumanMessage(user_prompt)]
    response = await self._llm.ainvoke(messages)
    content = response.content

    # 2. 提取 JSON（处理 LLM 输出中的额外文本）
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    plan_dict = json.loads(content[json_start:json_end])

    # 3. 构造 + 验证
    plan = ExecutionPlan(**plan_dict)
    plan = await self._validator.validate(plan)
    return plan
```

**异常处理**：如果 LLM 调用失败（网络错误、JSON 解析失败），自动回退到模板规划。

### 四种模板回退

| 模板 | 触发模式 | 生成结构 |
|------|---------|---------|
| **Pick-and-Place** | `pick up`, `grasp`, `grab` | perceive → navigate → locate → grasp (4步) |
| **Navigation** | `go to`, `navigate to` | perceive → navigate (2步) |
| **Search** | `find`, `look for`, `search` | survey → search_pattern (2步) |
| **Generic** | 其他 | perceive → speak_confirmation (2步) |

**Pick-and-Place 模板示例**：

```python
def _pick_and_place_plan(self, goal, task_spec) -> ExecutionPlan:
    # 用正则从 goal 中提取目标物体和房间
    object_match = re.search(r"(?:pick up|grasp|grab)\s+(\w+)", goal)
    room_match = re.search(r"(?:go to|in|from)\s+(\w+)", goal)

    sub_tasks = [
        SubTask("perceive_scene", SkillType.GAZE_AT, {"target": "scene"}),
        SubTask("navigate_to_{room}", SkillType.NAVIGATE_TO, {"room": room},
                preconditions=["perceive_scene"]),
        SubTask("locate_{object}", SkillType.GAZE_AT, {"target": obj},
                preconditions=["navigate_to_{room}"]),
        SubTask("grasp_{object}", SkillType.WHOLE_BODY_GRASP, {"object": obj, "arm": "right"},
                preconditions=["locate_{object}"]),
    ]
    return ExecutionPlan(sub_tasks=sub_tasks, estimated_duration_sec=90.0)
```

---

## 二、计划验证器 (`plan_validator.py`)

在规划生成后立即验证其可行性，确保机器人不会执行逻辑上矛盾的指令。

### 五项检查

| 检查项 | 说明 | 违反示例 |
|--------|------|---------|
| **唯一 ID** | 所有 sub_task_id 不能重复 | 两个相同的 "perceive_scene" |
| **DAG 无环** | 前置条件中不能有循环依赖 | A→B→C→A |
| **前置引用** | 所有 preconditions 必须引用存在的子任务 | 引用了不存在的 "make_coffee" |
| **合理超时** | timeout_sec ∈ (0, 600] | timeout=-1 或 timeout=1000 |
| **双足平衡** | 不能同时执行行走和操作 | stroll_while_carrying |

### DAG 环检测（DFS）

```python
@staticmethod
def _is_valid_dag(sub_tasks: list) -> bool:
    """使用 DFS + 递归栈检测环"""
    ids = {st.sub_task_id: st for st in sub_tasks}
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def _has_cycle(node_id: str) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)
        for precond in ids[node_id].preconditions:
            if precond not in visited:
                if _has_cycle(precond): return True
            elif precond in rec_stack:  # 后向边 → 发现环
                return True
        rec_stack.discard(node_id)
        return False

    for task_id in ids:
        if task_id not in visited:
            if _has_cycle(task_id): return False
    return True
```

### 双足平衡约束检查

```python
# 检测是否存在可并行的行走+操作组合
for i, st_a in enumerate(sub_tasks):
    for st_b in sub_tasks[i+1:]:
        if are_parallel(st_a, st_b):  # 两者无依赖关系，可能同时执行
            if is_walking(st_a) and is_manipulating(st_b):
                raise PlanValidationError("Cannot walk and manipulate simultaneously")
```

这个检查反映了人形机器人的核心物理约束：执行精细操作时必须有稳定的支撑（双足站立）。

---

## 三、运动规划器 (`motion_planner.py`)

**Phase 1 为 Stub**。预期实现：

```python
class MotionPlanner(AbstractPlanner):
    """全身运动规划器（ZMP 预览控制、脚步规划）"""
    # 输入：目标位姿、当前状态
    # 输出：关节轨迹 + 脚步序列
    # 方法：ZMP preview control, DDP, MPC
```

## 四、操作规划器 (`manipulation_planner.py`)

**Phase 1 为 Stub**。预期实现：

```python
class ManipulationPlanner(AbstractPlanner):
    """双臂操作规划器（IK求解、抓取生成）"""
    # 抓取类型：top_down_encompassing, cylindrical_encompassing,
    #           hook_grasp, bi_manual_pinch, bi_manual_support
    # 输出：手臂关节轨迹 + 夹爪指令序列
```

## 五、导航规划器 (`navigation_planner.py`)

**Phase 1 为 Stub**，但已实现了基础功能：

```python
class NavigationPlanner:
    def plan_path(self, start, goal) -> list[tuple]:
        """基于 BFS 的全局路径规划（直线距离回退）"""

    def navigate_to_room(self, current_room, target_room, spatial_memory) -> list[str]:
        """基于空间记忆中房间连通图的 BFS 搜索"""
        # 输入：current_room: "living_room", target_room: "kitchen"
        # 输出：["living_room", "corridor", "kitchen"]
```
