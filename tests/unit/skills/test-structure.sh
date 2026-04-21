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
# Test: Skill Structure
# =============================================================================
# Validates structure correctness for all skills.
# Rules tested:
# [Python validator — Test 1]
# - S-STR-01: YAML Front Matter format (---包裹)
# - S-STR-02: name field exists
# - S-STR-03: description field exists
# - S-STR-04: references/ directory not empty (if exists)
# - S-STR-05: name length 1-64 characters
# - S-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
# - S-STR-07: description length 1-1024; compatibility string 1-500
# - S-STR-09: File must be exactly SKILL.md (case-sensitive)
# - S-STR-10: Directory name must be kebab-case
# - S-STR-11: No README.md inside skill directory
# - S-STR-12: No XML angle brackets in frontmatter (security)
# - S-STR-13: SKILL.md body under 5000 words (warn-level, progressive disclosure)
# - S-STR-14: name has no reserved prefix (claude*/anthropic*)
# - S-STR-16: metadata is string->string mapping
# [Shell — Test 2]
# - S-STR-08: All links point to existing files (check_file_links)
# [Shell — Test 3]
# - S-STR-15: global name uniqueness (cross-repo)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

# Counters
total_skills=0
structure_pass=0
structure_fail=0
link_pass=0
link_fail=0

# Get all skills dynamically
ALL_SKILLS=$(get_all_skills)
total_skills=$(echo "$ALL_SKILLS" | wc -l)

echo "Found $total_skills skills"
echo ""

# ============================================
# Test 1: Skill Structure Validation
# ============================================
print_section_header "Test: Skill Structure (S-STR-01..07,09..14,16)"

for skill in $ALL_SKILLS; do
    skill_file=$(find_skill_file "$skill")
    
    if [ ! -f "$skill_file" ]; then
        print_skip "$skill: SKILL.md not found"
        continue
    fi
    
    if validate_skill_structure "$skill_file"; then
        ((structure_pass++)) || true
    else
        ((structure_fail++)) || true
    fi
done

echo ""

# ============================================
# Test 2: Link Validity
# ============================================
print_section_header "Test: Link Validity (S-STR-08)"

while IFS=: read -r sname spath; do
    if [ -f "$spath" ]; then
        if check_file_links "$spath" "skill"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done <<< "$(get_all_skills_with_paths)"

echo ""

# ============================================
# Test 3: Global Uniqueness (S-STR-15)
# ============================================
print_section_header "Test: Name Uniqueness (S-STR-15)"

uniq_pass=0
uniq_fail=0
if validate_global_uniqueness skill; then
    uniq_pass=1
else
    uniq_fail=1
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Skill Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total skills: $total_skills"
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