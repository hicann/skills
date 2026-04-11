# CANNBot PyPTO 算子开发快速入门指南

## 概述

CANNBot PyPTO 算子开发模式适用于通过 **PyPTO（Python Tensor Processing Operator）** 框架开发自定义算子。采用 7 阶段状态机驱动，覆盖从需求理解到性能调优的完整开发流程，支持断点续跑与失败恢复。

### 与 Kernel 直调开发的区别

| 对比维度 | PyPTO 算子开发（本模式） | Kernel 直调开发 |
|---------|------------------------|----------------|
| 适用场景 | PyPTO 框架算子开发 | Ascend C Kernel 直接开发 |
| 编程语言 | Python（PyPTO API） | C++（Ascend C API） |
| 开发内容 | PyPTO kernel + golden + test | Kernel + Tiling + Host 验证代码 |
| 阶段数 | 7 阶段（含 API 探索和 Golden 生成） | 7 步骤（含设计串讲和修复循环） |
| 精度验证 | 三态标记自动判定 | Reviewer 独立构建验证 |
| 性能调优 | 自动迭代调优（最多 10 轮） | Developer 手动采集 |

## 一、环境搭建

### 操作步骤

#### 方式一：项目级安装（推荐）

在项目目录下安装，配置仅对当前项目生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/skills.git

# 2. 进入 PyPTO 算子开发目录
cd skills/teams/pypto-op-orchestrator

# 3. 执行初始化脚本（项目级）
bash init.sh project opencode   # OpenCode 用户（默认）
bash init.sh project claude     # Claude Code 用户
```

#### 方式二：全局安装

在用户目录下安装，配置全局生效。

```bash
# 1. 克隆 CANN Skills 仓库
git clone https://gitcode.com/cann/skills.git

# 2. 进入 PyPTO 算子开发目录
cd skills/teams/pypto-op-orchestrator

# 3. 执行初始化脚本（全局）
bash init.sh global opencode    # OpenCode 用户（默认）
bash init.sh global claude      # Claude Code 用户
```

### 安装内容

init.sh 脚本会完成以下操作：

| 内容 | OpenCode 项目级 | OpenCode 全局 | Claude 项目级 | Claude 全局 |
|------|----------------|---------------|---------------|-------------|
| Skills 技能模块 | `.opencode/skills/` | `~/.config/opencode/skills/` | `.claude/skills/` | `~/.claude/skills/` |
| Agents 子代理 | `.opencode/agents/` | `~/.config/opencode/agents/` | `.claude/agents/` | `~/.claude/agents/` |
| AGENTS.md | `.opencode/AGENTS.md` | `~/.config/opencode/AGENTS.md` | `.claude/CLAUDE.md` | `~/.claude/CLAUDE.md` |

### 环境校验

执行完上述步骤后，检查目录结构是否符合以下规范：

**项目级安装**：
```
skills/teams/pypto-op-orchestrator/
├── .opencode/
│   ├── skills/                    # 技能模块
│   │   ├── pypto-intent-understand/
│   │   ├── pypto-api-explore/
│   │   ├── pypto-golden-generate/
│   │   ├── pypto-op-design/
│   │   ├── pypto-op-orchestratorelop/
│   │   ├── pypto-op-perf-tune/
│   │   ├── pypto-precision-debug/
│   │   └── pypto-precision-compare/
│   ├── agents/                    # 子代理
│   │   ├── pypto-op-analyst/
│   │   ├── pypto-op-orchestratoreloper/
│   │   └── pypto-op-perf-tuner/
│   ├── AGENTS.md                  # Agent 配置
│   └── cannbot-manifest.json      # 安装清单
├── init.sh                        # 初始化脚本
└── quickstart.md                  # 本文档
```

## 二、快速上手

### 启动

在初始化完成的目录下执行：

```bash
opencode
```

### 开发算子示例

在交互界面中输入算子开发需求，CANNBot 会自动启动 7 阶段流程：

```
帮我开发一个 softmax 算子，支持 float16 数据类型，shape 主要是 [1,128]、[4,2048]、[32,4096]
```

### 核心工作流

采用 7 阶段状态机，确保算子开发质量：

```
Stage 1: 需求理解 → Stage 2: API 探索 → Stage 3: Golden 生成
    → Stage 4: 设计 → Stage 5: 代码实现 → Stage 6: 精度修复（按需）
    → Stage 7: 性能调优
```

每个阶段完成门禁校验后才能进入下一阶段。支持断点续跑和失败恢复，详见 AGENTS.md。

### 产出物示例

PyPTO 算子开发模式下，CANNBot 会在 `custom/{operator}/` 目录下生成以下文件：

```
custom/softmax/
├── SPEC.md                    # 需求规格
├── API_REPORT.md              # API 可行性报告
├── DESIGN.md                  # 设计文档
├── softmax_golden.py          # PyTorch 参考实现
├── softmax_impl.py            # PyPTO kernel 实现
├── test_softmax.py            # 测试入口
├── README.md                  # 实现说明
├── .orchestrator_state.json   # 流程状态（自动维护）
└── history_version/           # 版本备份
```

## 三、可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-api-explore` | API 可行性探索与分析 | Stage 2 |
| `pypto-golden-generate` | Golden 参考实现生成 | Stage 3 |
| `pypto-op-design` | 算子设计方案生成 | Stage 4 |
| `pypto-op-orchestratorelop` | 算子代码实现与测试 | Stage 5 |
| `pypto-precision-debug` | 精度问题代码层排查 | Stage 6 |
| `pypto-precision-compare` | 精度中间结果对比分析 | Stage 6（辅助） |
| `pypto-op-perf-tune` | 算子性能分析与自动调优 | Stage 7 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-op-analyst` | Golden 生成与设计 | Stage 3-4 |
| `pypto-op-orchestratoreloper` | 代码实现与精度修复 | Stage 5-6 |
| `pypto-op-perf-tuner` | 性能分析与调优 | Stage 7 |

## 四、断点续跑与恢复

CANNBot 通过 `.orchestrator_state.json` 维护全局状态，支持：

| 场景 | 使用方式 |
|------|---------|
| 中断后继续 | 再次输入算子名，自动从上次中断处续跑 |
| 失败后重试 | 输入"继续开发 {算子名}"，从失败阶段恢复 |
| 查看状态 | 查看 `custom/{op}/.orchestrator_state.json` |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新技能模块？

重新执行 init.sh 即可，脚本会自动覆盖旧版本。

### Q: PyPTO 和 Kernel 直调模式如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO 算子开发 |
| 使用 Ascend C API 开发算子 | Kernel 直调开发 |
| 快速验证算子可行性 | PyPTO 算子开发 |
| 需要精细控制硬件资源 | Kernel 直调开发 |
| 原型开发和概念验证 | PyPTO 算子开发 |
| 生产级高性能算子 | Kernel 直调开发 |

---

## 总结

1. PyPTO 算子开发模式通过 7 阶段状态机实现端到端自动化
2. 环境搭建核心两步：克隆仓库 → 执行 init.sh
3. `opencode` / `claude` 是核心交互指令
4. 所有阶段通过门禁驱动，支持断点续跑与失败恢复
5. 产出物包含完整的参考实现、设计文档、PyPTO 实现和测试入口
