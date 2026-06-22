# RoboClaw 文档索引

RoboClaw 是一个**生产级人形机器人垂直领域 Agent 框架**，基于 LangGraph StateGraph 编排引擎，实现感知→反思→规划→执行→反馈→恢复的完整闭环。

## 文档目录

| 章节 | 说明 |
|------|------|
| [架构总览](architecture.md) | 整体架构设计、Agent 循环、技术栈选型 |
| [核心层](core-layer.md) | 配置系统、类型枚举、错误体系、DI 容器 |
| [数据模型](models.md) | Pydantic v2 数据模型：RobotState、Task/SubTask DAG、感知、技能、安全 |
| [记忆系统](memory-system.md) | 五层记忆架构：工作记忆→情景记忆→语义记忆→空间记忆→知识库 |
| [编排引擎](orchestration.md) | LangGraph StateGraph 构建、8 个节点、条件路由、子图 |
| [规划层](planning.md) | LLM 任务分解器、计划验证器、运动/操作/导航规划器 |
| [执行层](action.md) | 技能库（9 种内置技能）、运动执行器、操作执行器 |
| [故障恢复](recovery.md) | 责任链恢复、5 种策略、指数退避重试、安全监控器 |
| [API 层](api-layer.md) | FastAPI 路由、中间件、WebSocket 遥测、SSE 流 |
| [A2A 与通信](a2a-communication.md) | 多智能体协作协议、发现/协商/委托、ROS2/RabbitMQ 通信 |

## 项目结构

```
roboclaw/
├── core/              # 核心基础：config, types, errors, registry
├── models/            # 10 个 Pydantic v2 模型模块
├── perception/        # 感知层（视觉、音频、本体感觉、传感器融合）
├── planning/          # 规划层（任务分解、运动/操作/导航规划、验证）
├── action/            # 执行层（技能库、运动、操作、导航、语音）
├── memory/            # 五层记忆（工作/情景/语义/空间/知识库 + 上下文编译器）
├── orchestration/     # LangGraph 编排（状态定义、节点、边、图、子图）
├── recovery/          # 故障恢复（责任链、5 种策略、重试、安全监控）
├── api/               # FastAPI（路由、中间件、WebSocket）
├── a2a/               # A2A 协议（服务端、客户端、注册表、协商）
├── comm/              # 通信层（ROS2、RabbitMQ、WebSocket、MQTT）
├── tools/             # LangChain 工具（感知、记忆、技能、安全、知识）
configs/               # YAML 配置文件
docker/                # Docker Compose 基础设施
examples/              # 3 个演示脚本
tests/                 # 55 个测试（单元 + 集成）
```

## 快速开始

```bash
# 安装
conda create -n roboclaw python=3.11
conda activate roboclaw
pip install -e .

# 运行测试
python -m pytest tests/ -v

# 运行演示
python examples/demo_full_loop.py      # 完整 Agent 循环
python examples/demo_recovery.py       # 故障恢复链
python examples/demo_a2a_swarm.py      # 多 Agent 协作

# 启动 API 服务
uvicorn roboclaw.api.app:create_app --factory --reload
```
