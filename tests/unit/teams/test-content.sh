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
# Test: Team Content
# =============================================================================
# Validates content quality for all teams.
# Rules tested:
# - T-CON-01: directory naming format ^[a-z0-9]+(-[a-z0-9]+)*$
# - T-CON-02: description contains trigger keywords
# - T-CON-03: description contains trigger conditions (warning: REQUIRED: when to use)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_teams=0
pass_count=0
fail_count=0

# Get all teams dynamically
ALL_TEAMS=$(get_all_teams)
total_teams=$(echo "$ALL_TEAMS" | wc -l)

echo "Found $total_teams teams"
echo ""

# ============================================
# Validate all teams content
# ============================================
print_section_header "Test: Team Content (T-CON-01..03)"

for team in $ALL_TEAMS; do
    team_file=$(find_team_file "$team")
    
    if [ ! -f "$team_file" ]; then
        print_skip "$team: AGENTS.md not found"
        continue
    fi
    
    if validate_team_content "$team_file"; then
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
echo -e " ${BOLD}Team Content Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
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