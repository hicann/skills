# CANN Skills

## 项目概述

### 项目定位

CANN Skills 为 CANN 社区提供可复用的 Agent 技能模块。目前已实现 Ascend C 算子开发全流程覆盖（12 个 Skills），并成功应用于 Agent 实践 CANNBot。未来将拓展至 CANN 更多技术领域。

### 目标用户

- CANN 社区开发者
- 昇腾 NPU 平台 AI 应用开发者
- Ascend C 算子开发者
- 希望贡献 Skills/Agents 的社区贡献者

## 快速开始

### 方式一：脚本安装（推荐）

适用于 Ascend C 算子直调开发场景，自动安装 Skills、配置文件及 asc-devkit 工具包。

```bash
git clone https://gitcode.com/cann/skills.git
cd skills/teams/ascendc-kernel-dev-team
bash init.sh
```

详细说明见 [teams/ascendc-kernel-dev-team/quickstart.md](teams/ascendc-kernel-dev-team/quickstart.md)。

### 方式二：手动安装

仅安装 Skills 和 Agents，适用于自定义配置场景。

```bash
git clone https://gitcode.com/cann/skills.git
cd skills
mkdir -p .opencode/{skills,agents} && cp -r skills/* .opencode/skills/ && cp -r agents/* .opencode/agents/
```

> 如需全局安装，将 `.opencode` 替换为 `~/.config/opencode` 即可。

### 启动 CLI

```bash
opencode
```

## 项目架构设计

### 整体架构

```
skills/
├── skills/                      # 技能模块库
├── agents/                      # 子 Agent
├── teams/                       # 多 Agent 协同
│   └── ascendc-kernel-dev-team/ # AsendC kernel开发流程
└── tests/                       # 自动化测试框架
```

## Skills 技能库

| Skill | 功能 |
|-------|------|
| **ascendc-kernel-develop-workflow** | 7 阶段工作流：环境准备 → 需求分析 → 算子实现 → 测试 → 问题处理 → 总结 → 文档 |
| **ascendc-api-best-practices** | API 使用最佳实践、参数限制 |
| **ascendc-npu-arch** | NPU 架构知识、芯片型号映射 |
| **ascendc-docs-search** | API 文档索引 + 在线搜索 |
| **ascendc-env-check** | NPU 设备查询、CANN 环境验证 |
| **ascendc-tiling-design** | Tiling 设计方法论，按算子类别分类 |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析、Kernel 挂起排查 |
| **ascendc-ut-develop** | UT 开发与覆盖率增强 |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0/L1 用例生成 |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 |
| **ascendc-task-focus** | 任务聚焦，解决长任务"迷失在中间"问题 |

## Agents 智能代理

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-reviewer** | 代码检视专家，支持快速检视和全功能检视两种模式 |

## CANNbot 智能助手（Teams）

CANNbot 是基于 Teams 配置的主 Agent，整合 Skills 和 Agents，提供端到端的任务支持。当前已实现 **ascendc-kernel-dev-team**（算子开发场景）。

## 测试框架

自动化测试验证 Skills 和 Agents 的正确性，确保技能模块和智能代理的行为符合预期。
详见 [tests/README.md](tests/README.md)。

## 许可证

本项目遵循华为 CANN 社区许可证协议，详见 [LICENSE](LICENSE) 文件。

## 免责声明

感谢您关注 CANN Skills 项目！我们希望这些技能和知识能帮助您更好地进行 CANN 开发。

在使用之前，有几点需要您了解：

1. **关于内容质量**：由于技术更新迭代，部分内容可能无法完全适用于所有场景。如果发现问题，欢迎提 Issue 告诉我们。

2. **关于使用目的**：本仓库内容仅供技术参考和学习使用，建议在测试环境充分验证后再用于生产场景。
