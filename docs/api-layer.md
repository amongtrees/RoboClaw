# API 层 (FastAPI)

提供 REST API、WebSocket 实时遥测和 SSE 事件流。位于 `roboclaw/api/`。

---

## 一、应用工厂 (`app.py`)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时：初始化数据库连接池、Redis 客户端、安全监控器
    yield
    # 关闭时：优雅关闭所有连接

def create_app() -> FastAPI:
    app = FastAPI(
        title="RoboClaw Agent API",
        description="Humanoid Robotics Agent Framework",
        version="0.1.0",
        lifespan=lifespan,
    )
    # CORS 中间件
    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

    # 加载路由
    app.include_router(admin_router)
    app.include_router(agent_router)
    app.include_router(task_router)
    app.include_router(episode_router)
    app.include_router(telemetry_router)
    app.include_router(a2a_router)

    return app
```

---

## 二、六个路由组

### 2.1 Agent 路由 (`routes/agent.py`)

管理机器人 Agent 的全生命周期。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/agents` | POST | 注册新机器人 Agent |
| `/agents/{robot_id}` | GET | 查询 Agent 状态和模式 |
| `/agents/{robot_id}/mode` | PUT | 切换工作模式 |
| `/agents/{robot_id}` | DELETE | 注销 Agent |
| `/agents` | GET | 列出所有已注册 Agent |

**关键代码**：

```python
@router.post("/agents")
async def register_agent(body: dict):
    robot_id = body.get("robot_id", "")
    if not robot_id:
        raise HTTPException(status_code=400, detail="robot_id is required")
    agent_info = {
        "robot_id": robot_id,
        "robot_model": body.get("robot_model", "unknown"),
        "display_name": body.get("display_name", robot_id),
        "status": "idle",
        "mode": body.get("mode", "autonomous"),
    }
    _agents[robot_id] = agent_info  # Phase 1: 内存存储
    return agent_info

@router.put("/agents/{robot_id}/mode")
async def set_agent_mode(robot_id: str, body: dict):
    mode = body.get("mode", "autonomous")
    valid_modes = {"idle", "autonomous", "teleop", "diagnostic", "emergency_stop"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of: {valid_modes}")
    _agents[robot_id]["mode"] = mode
    return _agents[robot_id]
```

**五种工作模式**：
- `idle` — 空闲，不接受任务
- `autonomous` — 全自主执行
- `teleop` — 遥操作模式
- `diagnostic` — 诊断模式（自检）
- `emergency_stop` — 紧急停止

### 2.2 Task 路由 (`routes/task.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/tasks` | POST | 提交新任务 |
| `/tasks/{task_id}` | GET | 查询任务状态和当前阶段 |
| `/tasks/{task_id}/plan` | GET | 获取任务生成的执行计划 |
| `/tasks/{task_id}` | DELETE | 取消运行中的任务 |
| `/tasks` | GET | 列出任务（可按 robot_id/status 过滤） |

**错误处理**：
- 400：缺少 robot_id 或 goal
- 404：任务不存在
- 409：任务已处于终态（completed/failed/cancelled），无法取消

### 2.3 Episode 路由 (`routes/episode.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/episodes` | POST | 创建新情节记录 |
| `/episodes/{episode_id}` | GET | 获取情节详情（含事件列表） |
| `/episodes/search` | GET | 多条件搜索（robot_id, task_type, outcome, query_text） |
| `/episodes/{episode_id}/events` | POST | 向情节添加事件 |
| `/episodes/{episode_id}/complete` | POST | 标记情节为完成 |

### 2.4 Telemetry 路由 (`routes/telemetry.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/telemetry/{robot_id}/state` | GET | 获取最近一次状态快照 |
| `/telemetry/{robot_id}/stream` | GET | **SSE 流式推送**（Server-Sent Events） |

**SSE 流示例**：
```
GET /telemetry/h1_01/stream

event: state_update
data: {"robot_id":"h1_01","phase":"acting","zmp":[0.01,-0.02],"joints":{}}

event: safety_alert
data: {"type":"force_limit","severity":"warning","component":"left_arm"}
```

#### WebSocket 遥测

```python
class WebSocketManager:
    """管理所有 WebSocket 连接，支持双向通信"""
    async def connect(self, robot_id, websocket): ...
    async def disconnect(self, robot_id, websocket): ...
    async def broadcast_telemetry(self, robot_id, data): ...
    async def send_command(self, robot_id, command): ...
```

WebSocket 端点：

| 端点 | 说明 |
|------|------|
| `/ws/telemetry/{robot_id}` | **单向**：服务器以 10Hz 推送遥测数据 |
| `/ws/command/{robot_id}` | **双向**：接收控制指令，返回状态（支持 pause/resume/estop/hri_message） |

**命令协议**：
```json
// 客户端 → 服务器
{"command": "pause"}
{"command": "resume"}
{"command": "estop"}
{"command": "hri_message", "payload": {"text": "继续执行"}}

// 服务器 → 客户端
{"status": "paused", "timestamp": 1719600000.0}
{"status": "estop_active", "timestamp": 1719600001.0}
```

### 2.5 A2A 路由 (`routes/a2a_routes.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/.well-known/agent.json` | GET | Agent Card（遵循 Google A2A 规范） |
| `/a2a/tasks` | POST | 接收同伴 Agent 委托的任务 |
| `/a2a/tasks/{task_id}/status` | GET | 查询委托任务状态 |
| `/a2a/tasks/{task_id}/stream` | GET | SSE 流式推送委托任务进度 |

### 2.6 Admin 路由 (`routes/admin.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查（返回 200 + uptime） |
| `/ready` | GET | 就绪检查（检查数据库连接等） |
| `/metrics` | GET | 运行指标（请求数、延迟等） |

---

## 三、中间件 (`middleware/`)

### 请求日志中间件

```python
class LoggingMiddleware:
    async def __call__(self, request, call_next):
        start = time()
        response = await call_next(request)
        duration = time() - start
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
        return response
```

### 速率限制中间件

```python
class RateLimitMiddleware:
    def __init__(self, max_requests_per_minute=60):
        self._window: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, request, call_next):
        client_ip = request.client.host
        now = time()
        # 滑动窗口：只保留最近 60 秒内的请求
        self._window[client_ip] = [t for t in self._window[client_ip] if now - t < 60]
        if len(self._window[client_ip]) >= self.max_requests_per_minute:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        self._window[client_ip].append(now)
        return await call_next(request)
```

**实现细节**：
- 基于客户端 IP 的滑动窗口算法
- 内存实现（Phase 1），生产环境改用 Redis 分布式计数
- 超限返回 HTTP 429
