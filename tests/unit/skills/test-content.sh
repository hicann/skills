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
# Test: Skill Content
# =============================================================================
# Validates content quality for all skills.
# Rules tested:
# - S-CON-01: name matches directory name
# - S-CON-02: description contains trigger keywords (skipped if disable-model-invocation)
# - S-CON-03: description contains trigger conditions (skipped if disable-model-invocation)
# - S-CON-04: instructions are specific and actionable
# - S-CON-05: error handling / troubleshooting section exists
# - S-CON-06: usage examples provided (warning)
# - S-CON-07: progressive disclosure (long SKILL.md links to references/)
# - S-CON-08: description 三段式（动作+触发+关键词）(disable-model-invocation: 1 segment ok)
# - S-CON-09: description 无反模式短语
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_skills=0
pass_count=0
fail_count=0

# Get all skills dynamically
ALL_SKILLS=$(get_all_skills)
total_skills=$(echo "$ALL_SKILLS" | wc -l)

echo "Found $total_skills skills"
echo ""

# ============================================
# Validate all skills content
# ============================================
print_section_header "Test: Skill Content (S-CON-01..09)"

for skill in $ALL_SKILLS; do
    skill_file=$(find_skill_file "$skill")
    
    if [ ! -f "$skill_file" ]; then
        print_skip "$skill: SKILL.md not found"
        continue
    fi
    
    if validate_skill_content "$skill_file"; then
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
echo -e " ${BOLD}Skill Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills: $total_skills"
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