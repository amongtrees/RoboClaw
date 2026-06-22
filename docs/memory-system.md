# 五层记忆系统

人形机器人在真实环境中执行任务需要多层次的记忆。RoboClaw 实现了五级记忆架构，从毫秒级的工作记忆到持久化的知识库，全部带内存回退。

```
时效性          记忆层             后端                容量
─────────────────────────────────────────────────────
秒级    ← 工作记忆 (Working)      Redis Hash         当前状态 + 200事件环形缓冲
分钟级  ← 情景记忆 (Episodic)     PostgreSQL          全量历史任务执行记录
长期    ← 语义记忆 (Semantic)     Milvus 向量库       嵌入向量 + 相似检索
长期    ← 空间记忆 (Spatial)      内存度量语义地图      房间/物体/可导航性
长期    ← 知识库 (Knowledge)      PostgreSQL          结构化领域知识
                   ↑
          ContextCompiler (上下文编译器)
                   将五层记忆编译为 LLM 提示词
```

---

## 一、工作记忆 (`working_memory.py`)

**职责**：缓存当前任务执行周期的实时状态，是最快的记忆层。

### 数据结构

```python
class WorkingMemory:
    def __init__(self, robot_id, redis_client=None, max_events=200, state_ttl_s=10):
        # Redis 后端（可选）+ 进程内回退
        self._state: dict = {}              # 最新状态（双写 Redis + 内存）
        self._events: deque[maxlen=200]     # 事件环形缓冲
        self._task_context: dict = {}        # 当前任务上下文

    # Redis key 命名规范
    @property
    def _key(self) -> str:
        return f"roboclaw:{robot_id}:working_memory"
    @property
    def _events_key(self) -> str:
        return f"roboclaw:{robot_id}:events"
```

### 核心操作

| 方法 | 说明 | Redis 操作 |
|------|------|-----------|
| `update_state(**kwargs)` | 更新状态字段，自动添加时间戳 | `HSET` + `EXPIRE` (TTL) |
| `get_state()` | 获取完整当前状态 | `HGETALL` |
| `get_robot_state()` | 重建 `HumanoidState` 对象 | 同上，构造 Pydantic 模型 |
| `add_event(event_type, payload)` | 向环形缓冲添加事件 | `LPUSH` + `LTRIM` |
| `get_recent_events(n=50)` | 获取最近 N 个事件 | `LRANGE` |
| `set_task_context(task_id, ctx)` | 存储当前任务上下文 | `HSET` |
| `clear_task()` | 清理任务相关状态 | 清空 + `HDEL` |

### 关键设计

1. **TTL 自动过期**：工作记忆状态在 Redis 中设置 TTL (默认 10s)，避免僵尸数据
2. **事件环形缓冲**：`deque(maxlen=200)` + Redis `LTRIM` 双重保障，不无限增长
3. **优雅降级**：Redis 不可用时所有操作自动回退到进程内 dict，`try/except` 包裹所有 Redis 调用
4. **重构 HumanoidState**：`get_robot_state()` 从散落的 dict 字段重建类型安全的 `HumanoidState` 对象

---

## 二、情景记忆 (`episodic_memory.py`)

**职责**：持久化存储所有任务执行的历史记录（Episode），支持按条件检索、全文搜索。

### 数据表

```sql
-- episodes 表（主表）
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    robot_id TEXT NOT NULL,
    task_id TEXT,
    task_type TEXT,
    goal TEXT,
    skill_name TEXT,
    plan_id TEXT,
    started_at DOUBLE PRECISION,
    ended_at DOUBLE PRECISION,
    outcome TEXT,
    summary TEXT,
    initial_state_json JSONB,
    final_state_json JSONB,
    metadata_json JSONB,
    recovery_attempts INTEGER,
    total_duration_sec DOUBLE PRECISION
);
-- episode_events 表
-- artifacts 表
```

### 核心操作

```python
class EpisodicMemory:
    async def save_episode(episode: EpisodeRecord) -> str:
        """保存新 Episode，存在则更新 (UPSERT)"""
        # SQL: INSERT ... ON CONFLICT (episode_id) DO UPDATE

    async def get_episode(episode_id: str) -> EpisodeRecord | None:
        """按 ID 查询"""

    async def search_episodes(
        robot_id=None, task_type=None, outcome=None,
        query_text=None, limit=20, offset=0
    ) -> list[EpisodeRecord]:
        """多条件组合查询 + PostgreSQL 全文搜索"""
        # 使用 tsvector/plainto_tsquery 支持自然语言搜索

    # Events 和 Artifacts 的 CRUD...
```

### 关键设计

1. **全文搜索**：利用 PostgreSQL `tsvector` 对 `goal` 和 `summary` 进行英文全文搜索
2. **UPSERT 语义**：保存 Episode 使用 `ON CONFLICT DO UPDATE`，避免重复
3. **JSONB 存储**：`initial_state_json` 和 `final_state_json` 使用 JSONB 类型，支持结构化查询
4. **内存回退**：无数据库时使用进程内 dict，排序逻辑保持一致

---

## 三、语义记忆 (`semantic_memory.py`)

**职责**：基于向量相似度的语义检索，用于"找到类似的历史任务"、"匹配最佳技能"、"检索已知故障模式"。

### 三个 Collection

| Collection | 用途 | 典型查询 |
|-----------|------|---------|
| `episode_embeddings` | 历史任务情节的语义索引 | "类似 'pick up cup from table' 的任务" |
| `skill_embeddings` | 技能与任务描述的匹配 | "哪个技能适合 '把杯子递给人类'？" |
| `failure_patterns` | 已知故障模式的向量索引 | "当前错误和哪些历史故障最相似？" |

### 核心操作

```python
class SemanticMemory:
    async def insert(collection_name, entry_id, embedding, metadata) -> str:
        """插入向量（Milvus + 内存双写）"""

    async def search(collection_name, query_embedding, query_text="", top_k=10, filters=None) -> list:
        """相似检索（Milvus 优先，回退内存余弦相似度）"""

    # 便捷方法
    async def index_episode(episode_id, embedding, summary, task_type, outcome): ...
    async def find_similar_episodes(embedding, top_k=5): ...
    async def find_failure_patterns(embedding, top_k=5): ...
    async def find_skills(embedding, top_k=10): ...
```

### 余弦相似度实现（内存回退）

```python
@staticmethod
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """当 Milvus 不可用时，在内存中手动计算余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

### 关键设计

1. **内存回退**：当 `pymilvus` 不可用或连接失败时，所有向量在内存 list 中存储，检索时逐一计算余弦相似度
2. **Milvus 自动连接**：`initialize()` 调用时自动连接，失败则标记 `_connected = False`
3. **过滤表达式**：`_build_filter_expr()` 将 dict 过滤器转换为 Milvus 表达式字符串

---

## 四、空间记忆 (`spatial_memory.py`)

**职责**：维护环境的度量-语义地图，包括房间拓扑、物体位置、可导航区域。

### 数据结构

```python
class SpatialMemory:
    def __init__(self, resolution_m: float = 0.05):
        # 房间图（带边界信息）
        self._rooms: dict[str, dict] = {}

        # 房间连通性（门、走廊）
        self._room_connections: list[tuple[str, str, str]] = []

        # 物体位置
        self._object_locations: dict[str, dict] = {}

        # 可导航网格（2D 占据栅格）
        self._navigable_grid: dict[tuple[int, int], bool] = {}

        # 机器人自身信念
        self._current_room: str = "unknown"
        self._robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
```

### 核心操作

| 方法 | 说明 |
|------|------|
| `add_room(room_id, name, bounds)` | 注册新房间 |
| `connect_rooms(room_a, room_b, type)` | 连接两个房间（doorway / corridor） |
| `get_connected_rooms(room_id)` | 获取相邻房间（用于 BFS 路径规划） |
| `place_object(obj_id, label, position, room_id)` | 记录物体位置 |
| `find_object(label)` | 按标签查找物体 |
| `set_navigable(x, y, bool)` | 标记栅格可导航性 |
| `is_navigable(position)` | 查询某位置是否可导航 |
| `to_dict()` / `from_dict()` | 序列化/反序列化（用于持久化） |

### 关键设计

1. **度量-语义双层**：底层是栅格地图（可导航性），上层是语义图（房间名、物体标签）
2. **序列化支持**：`to_dict()` / `from_dict()` 支持将空间地图持久化到文件或数据库
3. **当前仅内存实现**：Phase 1 为内存实现，后续可扩展为 protobuf 持久化

---

## 五、知识库 (`knowledge_base.py`)

**职责**：存储结构化的领域知识事实，不依赖向量检索，通过精确 key 查询。

### 四个知识域

| 域 (domain) | 说明 | 示例 |
|------------|------|------|
| `object_affordance` | 物体可供性（如何抓取） | cup → top_down_encompassing, 3.0N |
| `safety_rule` | 硬性安全约束 | max_gripper_force: 15N, human_proximity_stop: 0.5m |
| `kinematic_limit` | 运动学限制 | arm_workspace_radius: 0.85m |
| `skill_parameter` | 技能默认参数 | whole_body_grasp: approach_speed=0.1m/s, timeout=5s |

### 15+ 预置知识条目

```python
# 物体可供性（6个）
("cup", {"grasp_type": "top_down_encompassing", "grip_force_n": 3.0})
("bottle", {"grasp_type": "cylindrical_encompassing", "grip_force_n": 5.0})
("door_handle", {"grasp_type": "hook_grasp", "grip_force_n": 10.0})
("box_small", {"grasp_type": "top_down_encompassing", "grip_force_n": 4.0})
("box_large", {"grasp_type": "bi_manual_pinch", "grip_force_n": 8.0, "requires_dual_arm": True})
("tray", {"grasp_type": "bi_manual_support", "grip_force_n": 2.0, "requires_dual_arm": True})

# 安全规则（5个）、运动学限制（2个）、技能参数（4个）...
```

### 核心操作

```python
async def query(domain: str, key: str) -> dict | None:
    """精确查询一条知识"""

async def query_all(domain: str) -> list[dict]:
    """获取某个域的所有条目"""

async def upsert(domain: str, key: str, value: dict):
    """插入或更新（PostgreSQL ON CONFLICT DO UPDATE）"""

# 便捷方法
async def get_affordance(object_label: str) -> dict
async def get_safety_rules() -> list[dict]
async def get_skill_params(skill_name: str) -> dict
```

---

## 六、上下文编译器 (`context_compiler.py`)

**职责**：将五层记忆采集并编译为结构化的 LLM 提示词上下文。这是让规划"变聪明"的关键桥梁。

### 编译流程

```
输入：robot_id, task_goal, current_state, query_embedding

Step 1: 工作记忆 → current_state_summary (当前机器人状态的文本摘要)
Step 2: 情景记忆 → relevant_episodes (过去 5 个相关任务的执行记录)
Step 3: 语义记忆 → relevant_semantic (向量检索 5 个最相似的任务)
Step 4: 知识库 → relevant_knowledge (extract object→affordance + safety rules)
Step 5: 空间记忆 → spatial_context (current_room, robot_position, room list)

输出：LLMContext (包含所有采集结果的结构化对象)
```

### 状态摘要生成

```python
@staticmethod
def _summarize_state(state: HumanoidState) -> str:
    """将 HumanoidState 转换为自然语言摘要"""
    # 输出示例：
    # Robot: h1_unitree
    # Phase: planning
    # Joint count: 28
    # ZMP: (0.012, -0.003)
    # Left EE: (0.35, 0.15, 1.10)
    # Right EE: (0.35, -0.15, 1.08)
    # Support: double
    # Objects visible: 3
    # Detected: cup, table, chair
```

### 在规划中的使用

```python
# TaskDecomposer.plan() 内部
context = await context_compiler.compile(
    robot_id="h1_01",
    task_goal="go to kitchen and pick up the cup",
    query_embedding=goal_embedding,
)
# context.to_prompt_text() 生成 LLM 提示词中的上下文段落
```
