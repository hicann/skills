#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# Test: Agent Content
# =============================================================================
# Validates content quality for all agents.
# Rules tested:
# - A-CON-01: name matches directory/file name
# - A-CON-02: description contains trigger keywords
# - A-CON-03: description contains trigger conditions (warning: REQUIRED: when to use)
# - A-CON-04: instructions are specific and actionable (warning)
# - A-CON-05: error handling / troubleshooting section exists (warning)
# - A-CON-06: usage examples provided (warning)
# - A-CON-07: progressive disclosure (long files link to references/) (warning)
# - A-CON-08: description 三段式（动作+触发+关键词）(warning)
# - A-CON-09: description 无反模式短语 (warning)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_agents=0
pass_count=0
fail_count=0

# Get all agents dynamically
ALL_AGENTS=$(get_all_agents)
total_agents=$(echo "$ALL_AGENTS" | wc -l)

echo "Found $total_agents agents"
echo ""

# ============================================
# Validate all agents content
# ============================================
print_section_header "Test: Agent Content (A-CON-01..09)"

for agent in $ALL_AGENTS; do
    agent_file=$(find_agent_file "$agent")
    
    if [ ! -f "$agent_file" ]; then
        print_skip "$agent: <name>.md not found"
        continue
    fi
    
    if validate_agent_content "$agent_file"; then
        ((pass_count++)) || true
    else
        ((fail_count++)) || true
    fi
done

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Agent Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total agents: $total_agents"
echo -e "  ${GREEN}Passed:${NC}       $pass_count"
echo -e "  ${RED}Failed:${NC}       $fail_count"
echo ""

if [ $fail_count -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed content checks."
    exit 1
else
    print_status_passed
    exit 0
fi