#!/usr/bin/env bash
# =============================================================================
# Test: Skill Trigger Correctness
# =============================================================================
# What this tests:
# 1. Knowledge Skills response accuracy
# 2. Debug Skills response accuracy
# 3. Tool Skills response accuracy
# 4. Negative tests (irrelevant prompts should not trigger skill content)
# 5. Trigger mechanism tests (explicit/invalid skill requests)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Behavior Test ==="
echo ""
echo "This test verifies skills behavior with Claude CLI."
echo "Estimated time: 2-3 minutes"
echo "Requires: Claude Code CLI"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Claude Code CLI not found"
    echo "Install Claude Code first: https://code.claude.com"
    exit 1
fi

TIMEOUT=25
pass_count=0
fail_count=0
skip_count=0

# =============================================================================
# Part 1: Knowledge Skills
# =============================================================================

echo -e "${BOLD}=== Part 1: Knowledge Skills ===${NC}"
echo ""

run_behavior_test "ascendc-npu-arch" \
    "Ascend910B 和 Ascend950 分别对应什么架构？" \
    "arch32|arch35|达芬奇|DaVinci|架构" \
    "$TIMEOUT"

run_behavior_test "ascendc-api-best-practices" \
    "Ascend C 中 DataCopy 的主要作用是什么？" \
    "搬运|copy|数据|传输" \
    "$TIMEOUT"

run_behavior_test "ascendc-operator-kernel-design" \
    "Ascend C Kernel 的核函数签名需要哪些关键修饰符？" \
    "__global__|__aicore__" \
    "$TIMEOUT"

run_behavior_test "ascendc-code-review" \
    "为什么要避免使用 GM_ADDR 直接计算地址？" \
    "对齐|align|越界|boundary|安全|风险|边界|类型检查|缓冲区" \
    "$TIMEOUT"

run_behavior_test "ascendc-custom-op-enhance" \
    "开发 Ascend C 自定义算子的基本步骤是什么？" \
    "设计|实现|测试|开发|编译|构建" \
    "$TIMEOUT"

# =============================================================================
# Part 2: Debug Skills
# =============================================================================

echo -e "${BOLD}=== Part 2: Debug Skills ===${NC}"
echo ""

run_behavior_test "ascendc-runtime-debug" \
    "算子运行时报错，我该怎么调试？" \
    "运行时|runtime|错误|error|调试|print|log|定位|plog" \
    "$TIMEOUT"

run_behavior_test "ascendc-precision-debug" \
    "算子精度不达标，应该怎么排查？" \
    "精度|precision|误差|对比|检查|数据" \
    "$TIMEOUT"

run_behavior_test "ascendc-perf-analysis" \
    "算子性能太慢，如何优化？" \
    "性能|performance|优化|optimize|瓶颈|并行" \
    "$TIMEOUT"

run_behavior_test "ascendc-env-check" \
    "如何检查 Ascend NPU 环境是否正常？" \
    "npu-smi|环境|检查|NPU|设备" \
    "$TIMEOUT"

# =============================================================================
# Part 3: Tool Skills
# =============================================================================

echo -e "${BOLD}=== Part 3: Tool Skills ===${NC}"
echo ""

run_behavior_test "ascendc-kernel-develop-workflow" \
    "Ascend C Kernel 开发的第一阶段是什么？" \
    "环境|准备|phase|分析|设计|需求|Kernel" \
    "$TIMEOUT"

run_behavior_test "ascendc-ut-develop" \
    "Ascend C UT 测试应该放在哪个目录？" \
    "tests/ut|ut/|ut 目录|unit/|单元测试|tests/" \
    "$TIMEOUT"

# =============================================================================
# Part 4: Negative Tests
# =============================================================================

echo -e "${BOLD}=== Part 4: Negative Tests ===${NC}"
echo ""

run_behavior_test "weather (should not trigger skill)" \
    "今天天气怎么样？" \
    "天气|weather|晴|雨|温度|今天" \
    "$TIMEOUT"

run_behavior_test "poetry (should not trigger skill)" \
    "帮我写一首诗" \
    "诗|诗歌|韵律|押韵|春风|夏雨|明月|山河|云飞" \
    "$TIMEOUT"

run_behavior_test "cooking (should not trigger skill)" \
    "如何做一道红烧肉？" \
    "红烧|肉|烹饪|做法|食材|调料|锅|火" \
    "$TIMEOUT"

# =============================================================================
# Part 5: Trigger Mechanism Tests
# =============================================================================

echo -e "${BOLD}=== Part 5: Trigger Mechanism Tests ===${NC}"
echo ""

run_behavior_test "explicit skill request" \
    "请使用 ascendc-runtime-debug 技能分析运行时错误。" \
    "运行时|runtime|错误|error|调试" \
    "$TIMEOUT"

run_behavior_test "invalid skill handling" \
    "请使用 nonexistent-skill-xyz 技能。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到" \
    "$TIMEOUT"

run_behavior_test "skill name typo handling" \
    "请使用 ascendc-runtime-debub 技能。" \
    "不存在|not.*found|无法|unknown|没有|抱歉|找不到|运行时|runtime" \
    "$TIMEOUT"

# =============================================================================
# Summary
# =============================================================================

echo "========================================"
echo -e " ${BOLD}Skill Behavior Test Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $pass_count"
echo -e "  ${RED}Failed:${NC}  $fail_count"
echo -e "  ${YELLOW}Skipped:${NC} $skip_count"
echo ""

total=$((pass_count + fail_count))
if [ $total -gt 0 ]; then
    accuracy=$((pass_count * 100 / total))
    echo "  Accuracy: ${accuracy}%"
    echo ""
fi

if [ $fail_count -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi