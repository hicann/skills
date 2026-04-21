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
# Test: Agent Structure
# =============================================================================
# Validates structure correctness for all agents.
# Rules tested:
# [Python validator — Test 1]
# - A-STR-01: YAML Front Matter format
# - A-STR-02: name field exists (via _name_checks); mode field exists (shared rule ID)
# - A-STR-03: description field exists (via _description_checks); mode must be primary/subagent (shared rule ID)
# - A-STR-04: All skill dependencies exist
# - A-STR-05: name length 1-64 characters (via _name_checks)
# - A-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$ (via _name_checks)
# - A-STR-07: description length 1-1024 characters (via _description_checks)
# - A-STR-14: name has no reserved prefix (via _name_checks)
# [Shell — Test 2]
# - A-STR-08: All links point to existing files (check_file_links)
# [Shell — Test 3]
# - A-STR-09: global name uniqueness (cross-repo)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_agents=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0

# Get all agents dynamically
ALL_AGENTS=$(get_all_agents)
total_agents=$(echo "$ALL_AGENTS" | wc -l)

echo "Found $total_agents agents"
echo ""

# ============================================
# Test 1: Agent Structure Validation
# ============================================
print_section_header "Test: Agent Structure (A-STR-01..07,14)"

for agent in $ALL_AGENTS; do
    agent_file=$(find_agent_file "$agent")
    
    if [ ! -f "$agent_file" ]; then
        print_skip "$agent: <name>.md not found"
        continue
    fi
    
    if validate_agent_structure "$agent_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (A-STR-08)"

while IFS=: read -r aname apath; do
    if [ -f "$apath" ]; then
        if check_file_links "$apath" "agent"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_agents_with_paths)"

echo ""

# ============================================
# Test 3: Global Uniqueness (A-STR-09)
# ============================================
print_section_header "Test: Name Uniqueness (A-STR-09)"

uniq_pass=0
uniq_fail=0
if validate_global_uniqueness agent; then
    uniq_pass=1
else
    uniq_fail=1
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Agent Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total agents: $total_agents"
echo -e "  Structure tests: ${GREEN}$structure_pass passed${NC}, ${RED}$structure_fail failed${NC}"
echo -e "  Link tests:      ${GREEN}$link_pass passed${NC}, ${RED}$link_fail failed${NC}"
echo -e "  Uniqueness:      ${GREEN}$uniq_pass passed${NC}, ${RED}$uniq_fail failed${NC}"
echo ""

if [ $((structure_fail + link_fail + uniq_fail)) -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed structure checks."
    exit 1
else
    print_status_passed
    exit 0
fi