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
# Test: Team Structure
# =============================================================================
# Validates structure correctness for all teams.
# Rules tested:
# [Python validator — Test 1]
# - T-STR-01: YAML Front Matter format (---wrapped)
# - T-STR-02: mode field exists; mode must be "primary" (shared rule ID)
# - T-STR-03: description field exists (via _description_checks); skills field exists (shared rule ID)
# - T-STR-04: All skill dependencies exist
# - T-STR-05: references/ directory non-empty (only checked if directory exists)
# - T-STR-07: description length 1-1024 (via _description_checks; also used by uniqueness check)
# [Shell — Test 2]
# - T-STR-06: All links point to existing files (check_file_links)
# [Shell — Test 3]
# - T-STR-07: global name uniqueness (cross-repo; shared rule ID with T-STR-01..05)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_teams=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0

# Get all teams dynamically
ALL_TEAMS=$(get_all_teams)
total_teams=$(echo "$ALL_TEAMS" | wc -l)

echo "Found $total_teams teams"
echo ""

# ============================================
# Test 1: Team Structure Validation
# ============================================
print_section_header "Test: Team Structure (T-STR-01..05,07)"

for team in $ALL_TEAMS; do
    team_file=$(find_team_file "$team")
    
    if [ ! -f "$team_file" ]; then
        print_skip "$team: AGENTS.md not found"
        continue
    fi
    
    if validate_team_structure "$team_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (T-STR-06)"

while IFS=: read -r tname tpath; do
    if [ -f "$tpath" ]; then
        if check_file_links "$tpath" "team"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_teams_with_paths)"

echo ""

# ============================================
# Test 3: Global Uniqueness (T-STR-07)
# ============================================
print_section_header "Test: Name Uniqueness (T-STR-07)"

uniq_pass=0
uniq_fail=0
if validate_global_uniqueness team; then
    uniq_pass=1
else
    uniq_fail=1
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Team Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
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