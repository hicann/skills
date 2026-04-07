# CANNBot Skills

## 项目概述

### 项目定位

**CANNBot** 是面向CANN开发的用于提升开发效率的系列智能体，本仓库为其提供可复用的Skills模块，目前已实现 Ascend C 算子开发全流程覆盖，未来将拓展至 CANN 更多技术领域。

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
cd skills/teams/ops-direct-invoke
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
```

详细说明见 [teams/ops-direct-invoke/quickstart.md](teams/ops-direct-invoke/quickstart.md)。

### 方式二：手动安装

仅安装 Skills 和 Agents，适用于自定义配置场景。

```bash
git clone https://gitcode.com/cann/skills.git
cd skills
# OpenCode 用户
mkdir -p .opencode && ln -s ../skills .opencode/skills && ln -s ../agents .opencode/agents
# Claude 用户：将 .opencode 替换为 .claude
```

> 如需全局安装，OpenCode 用户将 `.opencode` 替换为 `~/.config/opencode`，Claude 用户替换为 `~/.claude`。

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
│   └── ops-direct-invoke/      # 算子直调开发流程
└── tests/                       # 自动化测试框架
```

### 逻辑架构视图

项目遵循三层架构：Teams 编排 Agents，Agents 绑定 Skills。以下视图展示各层组件及其关联关系。

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TEAMS（应用编排层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────┐        ┌─────────────────────────────┐      ║
║  │  ops-direct-invoke          │        │  ascendc-ops-dev-team ⬆    │      ║
║  │  Kernel 直调开发流程         │        │  自定义算子开发流程           │      ║
║  └──────┬──────┬──────┬────────┘        └──────┬──────┬──────┬────────┘      ║
║         │      │      │                        │      │      │               ║
╚═════════╪══════╪══════╪════════════════════════╪══════╪══════╪═══════════════╝
          │      │      │                        │      │      │
          ▼      ▼      ▼                        ▼      ▼      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             AGENTS（角色执行层）                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ║
║  │  architect   │  │  developer ⬆│  │  reviewer    │  │  tester ⬆   │      ║
║  │   方案设计    │  │   代码开发   │  │   代码检视    │  │   代码测试    │      ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
             │ │ │ │       │ │ │ │       │ │ │ │       │ │ │ │
             ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                             SKILLS（知识能力层）                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─ 知识库类 ──────────────────────────────────────────────────────────────┐  ║
║  │  npu-arch             NPU 架构知识与芯片映射                             │  ║
║  │  tiling-design        Tiling 设计方法论                                 │  ║
║  │  api-best-practices   API 使用最佳实践                                   │  ║
║  │  ops-precision-standard 算子精度标准                                     │  ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─ 工程模板类 ────────────────────────────────────────────────────────────┐  ║
║  │  registry-invoke-to-direct-invoke  注册算子直调改造模板                   │   ║
║  │  direct-invoke-template            Kernel直调工程模板                     │   ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  ┌─ 调试与测试类 ──────────────────────────────────────────────────────────┐  ║
║  │  precision-debug      精度调试与症状速查                                 │   ║
║  │  runtime-debug        运行时错误码解析                                   │  ║
║  │  ops-profiling        算子性能采集分析                                   │   ║
║  └────────────────────────────────────────────────────────────────────────┘  ║ 
║                                                                               ║
║  ┌─ 测试开发类 ────────────────────────────────────────────────────────────┐  ║
║  │  st-design            ST 测试用例设计                                   │   ║
║  │  ut-develop           UT 开发与覆盖率增强                               │   ║
║  │  code-review          代码检视规则                                      │  ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─ 工具辅助类 ────────────────────────────────────────────────────────────┐   ║
║  │  env-check            NPU 设备查询与环境验证                             │  ║
║  │  task-focus           长任务聚焦防迷失                                   │   ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

> 标注 ⬆ 的组件在规划中，后续会在社区上线

## Skills 技能库

| Skill | 功能 |
|-------|------|
| **ascendc-api-best-practices** | API 使用最佳实践、参数限制 |
| **ascendc-npu-arch** | NPU 架构知识、芯片型号映射 |
| **ascendc-docs-search** | API 文档索引 + 在线搜索 |
| **ascendc-env-check** | NPU 设备查询、CANN 环境验证 |
| **ascendc-tiling-design** | Tiling和Kernel 设计方法论，按算子类别分类 |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析、Kernel 挂起排查 |
| **ascendc-ut-develop** | UT 开发与覆盖率增强 |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0/L1 用例生成 |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 |
| **ascendc-task-focus** | 任务聚焦，解决长任务"迷失在中间"问题 |
| **ascendc-whitebox-design** | 白盒测试用例生成 |
| **ascendc-registry-invoke-to-direct-invoke** | 注册调用算子转 `<<<>>>` kernel 直调 |
| **ascendc-direct-invoke-template** | Kernel 直调工程模板，提供验证过的样例工程和修改指南 |
| **ops-profiling** | NPU 性能采集与分析，CSV 指标解读、瓶颈定位、优化建议 |
| **ops-precision-standard** | 算子精度标准，按 dtype 分类提供 atol/rtol 精度比对标准 |

## Agents 智能代理

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-reviewer** | 代码检视专家，支持快速检视和全功能检视两种模式 |
| **ascendc-kernel-architect** | Kernel直调架构师，支持需求分析、API验证、方案设计 |
| **ascendc-kernel-developer** | Kernel直调开发者，支持代码实现、编译测试、性能采集、文档编写 |
| **ascendc-kernel-reviewer** | Kernel直调审查者，支持独立构建验证、7维度评分、精度验证 |

## 测试框架

自动化测试验证 Skills 和 Agents 的正确性，确保技能模块和智能代理的行为符合预期。
详见 [tests/README.md](tests/README.md)。

## 许可证

本项目遵循华为 CANN 社区许可证协议，详见 [LICENSE](LICENSE) 文件。

## 免责声明

感谢您关注 CANNBot Skills 项目！我们希望这些技能和知识能帮助您更好地进行 CANN 开发。

在使用之前，有几点需要您了解：

1. **关于内容质量**：由于技术更新迭代，部分内容可能无法完全适用于所有场景。如果发现问题，欢迎提 Issue 告诉我们。

2. **关于使用目的**：本仓库内容处于 experimental 状态，仅供技术参考和学习使用，建议在测试环境充分验证后再用于生产场景。
