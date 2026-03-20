#!/usr/bin/env bash
# =============================================================================
# Test: Skill Premature Action Detection
# =============================================================================
# Verifies that tools are not invoked BEFORE Skills are loaded.
# This ensures Claude follows the skill instructions before taking actions.
#
# What this tests:
# 1. Skill should be invoked before any Write/Edit/Bash operations
# 2. TodoWrite before Skill is acceptable (for planning)
# 3. Read before Skill is acceptable (for understanding context)
# 4. No Write/Edit operations should happen before Skill is loaded
#
# Usage: ./test-premature-action.sh [skill-name]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

SKILL_NAME="${1:-ascendc-kernel-develop-workflow}"
TIMEOUT="${2:-120}"

echo "=== Test: Skill Premature Action Detection ==="
echo ""
echo "This test verifies that tools are not invoked before Skills are loaded."
echo "Target skill: $SKILL_NAME"
echo "Timeout: ${TIMEOUT}s"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Claude Code CLI not found"
    exit 1
fi

TIMESTAMP=$(date +%s)
OUTPUT_DIR="/tmp/cann-skills-tests/${TIMESTAMP}/premature-action/${SKILL_NAME}"
mkdir -p "$OUTPUT_DIR"

# Test scenarios with prompts that might trigger premature actions
declare -A TEST_SCENARIOS=(
    ["debug_scenario"]="我的算子运行时报错了，错误码是 161001，请帮我分析一下原因并给出解决方案。"
    ["develop_scenario"]="我需要开发一个 Add 算子，请告诉我第一步应该做什么。"
    ["optimize_scenario"]="我的算子性能太差，运行时间超过预期，请帮我优化。"
    ["precision_scenario"]="算子精度测试不通过，输出结果与期望值偏差很大，怎么排查？"
)

test_premature_action() {
    local scenario="$1"
    local prompt="$2"
    local log_file="$OUTPUT_DIR/${scenario}.log"

    echo -e "${BOLD}--- Scenario: $scenario ---${NC}"
    echo "Prompt: ${prompt:0:50}..."
    echo ""

    # Run Claude and capture output
    # Note: Session analysis uses the most recent session from ~/.claude/projects/
    if timeout "$TIMEOUT" claude -p "$prompt" \
        --dangerously-skip-permissions \
        > "$log_file" 2>&1; then
        :
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_warn "Timeout after ${TIMEOUT}s"
        fi
    fi

    # Find the session file created by this interaction
    local session_file=$(find_recent_session 2)

    if [ -n "$session_file" ] && [ -f "$session_file" ]; then
        cp "$session_file" "$OUTPUT_DIR/${scenario}.jsonl"
        analyze_premature_actions "$session_file" "skill" "$SKILL_NAME"
    else
        print_warn "No session file found for analysis"
        print_info "Output saved to: $log_file"
    fi

    rm -f "$session_file" 2>/dev/null || true
    echo ""
}

# Run tests for each scenario
echo "Running premature action tests..."
echo ""

total_passed=0
total_failed=0

for scenario in "${!TEST_SCENARIOS[@]}"; do
    prompt="${TEST_SCENARIOS[$scenario]}"
    if test_premature_action "$scenario" "$prompt"; then
        ((total_passed++)) || true
    else
        ((total_failed++)) || true
    fi
done

# Summary
echo "========================================"
echo -e " ${BOLD}Premature Action Test Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $total_passed"
echo -e "  ${RED}Failed:${NC}  $total_failed"
echo ""
echo "  Output directory: $OUTPUT_DIR"
echo "    - *.log: CLI output logs"
echo "    - *.jsonl: Session transcripts (for analysis)"
echo ""

if [ $total_failed -gt 0 ]; then
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi