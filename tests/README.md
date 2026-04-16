# CANN Skills 测试框架

自动化测试框架，验证 skills 和 agents 的正确加载和行为。

## 要求

- Claude Code CLI 或 OpenCode CLI
- Bash 4.0+
- Python 3.6+ (用于 Token 分析)
- `jq` (可选，用于高级会话分析功能，如成本分析和工作流分析)

## 快速开始

```bash
# 运行单元测试（无需 CLI）
./run-tests.sh --fast

# 运行行为测试
./run-tests.sh --category behavior

# 运行全量测试
./run-tests.sh --integration

# 查看可用测试
./run-tests.sh --list

# 运行指定测试
./run-tests.sh --test unit/skills/test-structure.sh
./run-tests.sh --test behavior/agents/test-trigger-correctness.sh
```

## 目录结构

```
tests/
├── unit/                      # L1 单元测试（无需 CLI，< 30s）
│   ├── skills/
│   │   ├── test-structure.sh  # Skill 结构验证
│   │   └── test-content.sh    # Skill 内容验证
│   └── agents/
│       ├── test-structure.sh  # Agent 结构验证
│       └── test-content.sh    # Agent 内容验证
│
├── behavior/                  # L2 行为测试（需要 CLI，1-5 min）
│   ├── skills/
│   │   ├── test-trigger-correctness.sh
│   │   └── test-premature-action.sh
│   └── agents/
│       ├── test-trigger-correctness.sh
│       └── test-premature-action.sh
│
├── integration/               # L3 集成测试（5-15 min）
│   ├── test-simple-op-development.sh
│   └── test-workflow-execution.sh
│
├── lib/
│   └── test-helpers.sh        # 测试辅助函数
│
├── tools/
│   ├── analyze-session.sh
│   ├── analyze-tokens.sh
│   ├── analyze-workflow.sh
│   └── analyze-token-usage.py
│
└── run-tests.sh
```

## 测试分层

| 层级 | 目录 | 说明 | 运行方式 |
|------|------|------|----------|
| L1 | unit/ | 单元测试，验证结构和内容 | `--fast` |
| L2 | behavior/ | 行为测试，验证触发和响应 | `--category behavior` |
| L3 | integration/ | 集成测试，验证完整工作流 | `--integration` |

## 测试内容

### L1 单元测试

#### Skills 测试规则详情

| 规则ID | 测试项 | 级别 | 文件 |
|--------|-------|------|------|
| S-STR-01 | YAML格式正确（---包裹） | 必须 | test-structure.sh |
| S-STR-02 | name字段存在 | 必须 | test-structure.sh |
| S-STR-03 | description字段存在 | 必须 | test-structure.sh |
| S-STR-04 | references目录非空（如存在） | 条件 | test-structure.sh |
| S-STR-05 | name长度1-64字符 | 必须 | test-structure.sh |
| S-STR-06 | name格式正则验证 | 必须 | test-structure.sh |
| S-STR-07 | description长度1-1024字符 | 必须 | test-structure.sh |
| S-STR-08 | 链接有效 | 必须 | test-structure.sh |
| S-CON-01 | name与目录名一致 | 必须 | test-content.sh |
| S-CON-02 | description包含触发关键词 | 必须 | test-content.sh |
| S-CON-03 | description包含触发条件 | 建议 | test-content.sh |
| S-CON-04 | 命名符合前缀规范 | 必须 | test-content.sh |

#### Agents (unit/agents/)

| 测试文件 | 验证项 |
|---------|--------|
| `test-structure.sh` | YAML格式、name/description/mode字段、skills依赖、name/description格式、链接有效性 |
| `test-content.sh` | name一致性、description触发关键词、命名前缀、核心职责 |

#### Agents 测试规则详情

| 规则ID | 测试项 | 级别 | 文件 |
|--------|-------|------|------|
| A-STR-01 | YAML格式正确 | 必须 | test-structure.sh |
| A-STR-02 | name/description/mode字段存在 | 必须 | test-structure.sh |
| A-STR-03 | mode为primary或subagent | 必须 | test-structure.sh |
| A-STR-04 | skills依赖全部存在 | 必须 | test-structure.sh |
| A-STR-05 | name长度1-64字符 | 必须 | test-structure.sh |
| A-STR-06 | name格式正则验证 | 必须 | test-structure.sh |
| A-STR-07 | description长度1-1024字符 | 必须 | test-structure.sh |
| A-STR-08 | 链接有效 | 必须 | test-structure.sh |
| A-CON-01 | name与目录名一致 | 必须 | test-content.sh |
| A-CON-02 | description包含触发关键词 | 必须 | test-content.sh |
| A-CON-03 | 命名符合前缀规范 | 必须 | test-content.sh |
| A-CON-04 | 核心职责章节 | 必须 | test-content.sh |

#### Teams (unit/teams/)

| 测试文件 | 验证项 |
|---------|--------|
| `test-structure.sh` | YAML格式、description/mode/skills字段、依赖存在、description长度、链接有效性 |
| `test-content.sh` | 目录命名格式、触发关键词、核心原则章节 |
| `test-version.sh` | plugin.json SemVer 格式、Skill/Agent 变更检测、版本升级建议 |

#### Teams 版本看护规则

| 版本位 | 触发条件 | 示例 |
|--------|---------|------|
| **PATCH** (第3位) | Skills 列表或内容发生变化 | `1.0.0` → `1.0.1` |
| **MINOR** (第2位) | Agents 列表或内容发生变化 | `1.0.0` → `1.1.0` |
| **MAJOR** (第1位) | 团队工作流/接口不兼容变更 | `1.0.0` → `2.0.0` |

版本快照存储在 `tests/.version-state/<team-name>.json`，每次运行测试自动更新（仅 PASS 时）。

#### 市场注册表一致性

`plugin.json` 是 plugin 的权威定义，但用户通过市场（`package.json` / `marketplace.json`）看到的版本号决定是否需要升级。如果两者不同步，用户无法感知版本变化。

测试会在以下情况拦截：
- 修改了 `plugin.json` 的 version，但未同步更新 `package.json`（OpenCode）或 `marketplace.json`（Claude）

### L2 行为测试

#### Skills (behavior/skills/)

| 测试文件 | 说明 |
|---------|------|
| `test-trigger-correctness.sh` | 知识/调试/工具类技能触发测试 + 负向测试 |
| `test-premature-action.sh` | 验证 Skill 加载前无 Write/Edit/Bash 操作 |

#### Agents (behavior/agents/)

| 测试文件 | 说明 |
|---------|------|
| `test-trigger-correctness.sh` | 4 个 Agent 角色触发测试 + 负向测试 |
| `test-premature-action.sh` | 验证 Agent 调度前无不当操作 |

### L3 集成测试

| 测试文件 | 说明 |
|---------|------|
| `test-simple-op-development.sh` | 知识验证：文件结构、TilingData、Kernel 签名、芯片架构、ACLNN、开发流程、UT 测试 |
| `test-workflow-execution.sh` | 真实工作流：创建文件并验证内容 |

## 运行参数

| 参数 | 说明 |
|------|------|
| `--fast`, `-f` | 仅运行单元测试 |
| `--integration`, `-i` | 包含集成测试 |
| `--all` | 运行所有测试 |
| `--category`, `-c CAT` | 运行指定类别（unit/behavior/integration/all） |
| `--platform PLATFORM` | 指定平台（claude/opencode/auto，默认: opencode） |
| `--test`, `-t NAME` | 运行指定测试 |
| `--timeout SECONDS` | 设置超时时间（默认: 300） |
| `--verbose`, `-v` | 显示详细输出 |
| `--output FORMAT` | 输出格式（text/json） |
| `--list`, `-l` | 列出所有可用测试 |
| `--help`, `-h` | 显示帮助信息 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `NO_COLOR` | 设置后禁用彩色输出 |
| `FORCE_COLOR` | 设置为 `1` 强制启用彩色输出（适用于 CI/非 TTY 环境） |

## 测试辅助库

### test-helpers.sh

```bash
# 平台检测
is_platform_available "claude"
get_platform_version "claude"
detect_platforms

# 执行函数
run_claude "prompt" [timeout] [allowed_tools]
run_opencode "prompt" [timeout]
run_ai "prompt" [timeout] [platform]

# 断言函数
assert_contains "output" "pattern" "test name"
assert_not_contains "output" "pattern" "test name"
assert_count "output" "pattern" expected "test name"
assert_order "output" "pattern_a" "pattern_b" "test name"
assert_file_exists "/path/to/file" "test name"

# 查询函数
get_all_skills
get_all_agents

# 结构验证函数
validate_skill_structure "/path/to/SKILL.md"    # S-STR-01 to S-STR-08
validate_skill_content "/path/to/SKILL.md"      # S-CON-01 to S-CON-04
validate_agent_structure "/path/to/AGENT.md"    # A-STR-01 to A-STR-08
validate_agent_content "/path/to/AGENT.md"      # A-CON-01 to A-CON-04

# Session 分析
find_recent_session [minutes_old]
verify_skill_invoked "$session_file" "skill-name"
verify_agent_dispatched "$session_file" "agent-name"
count_tool_invocations "$session_file" "ToolName"
check_premature_action "$session_file" "skill-name"
get_triggered_skills "$session_file"
analyze_workflow_sequence "$session_file"
analyze_tool_chain "$session_file"
analyze_cost_breakdown "$session_file"
extract_token_usage "$session_file"

# 测试项目管理
create_test_project [prefix]
cleanup_test_project "$test_dir"

# 结果跟踪
init_test_tracking
record_test "pass" "test_name" ["duration"]
print_test_summary
output_test_json
```

## 分析工具

```bash
# Session 分析
./tools/analyze-session.sh session.jsonl --brief
./tools/analyze-session.sh session.jsonl --full
./tools/analyze-session.sh session.jsonl --json
./tools/analyze-session.sh session.jsonl --tools
./tools/analyze-session.sh session.jsonl --cost
```

### analyze-session.sh 选项

| 选项 | 说明 |
|------|------|
| `--brief` | 显示简要摘要（默认） |
| `--full` | 显示完整分析报告 |
| `--json` | 以 JSON 格式输出 |
| `--tools` | 显示工具调用详情 |
| `--cost` | 显示成本分析（需要 jq） |

### 其他工具

```bash
# Token 分析
./tools/analyze-tokens.sh session.jsonl

# 工作流分析
./tools/analyze-workflow.sh session.jsonl

# Python Token 分析
python3 tools/analyze-token-usage.py session.jsonl
```

## 添加新测试

### 单元测试

在 `unit/skills/` 或 `unit/agents/` 创建测试文件：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Your Test Name ==="

if [ condition ]; then
    echo "  [PASS] Test description"
else
    echo "  [FAIL] Test description"
    exit 1
fi
```

### 行为测试

在 `behavior/skills/` 或 `behavior/agents/` 创建测试文件：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

TIMEOUT=30
output=$(run_claude "your prompt" $TIMEOUT)

if echo "$output" | grep -qiE "expected_pattern"; then
    echo "  [PASS] Test passed"
else
    echo "  [FAIL] Test failed"
fi
```

## 故障排查

### CLI 未找到

```bash
npm install -g @anthropic/claude-code
```

### 测试超时

```bash
./run-tests.sh --timeout 900
```

### 无 CLI 环境

```bash
./run-tests.sh --fast
```

### 查看详细日志

```bash
./run-tests.sh --verbose --test unit/skills/test-structure.sh
```