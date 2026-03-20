#!/usr/bin/env bash
# =============================================================================
# Test: Skill Structure
# =============================================================================
# Validates structure correctness for all skills.
# Rules tested:
# - S-STR-01: YAML Front Matter format (---包裹)
# - S-STR-02: name field exists
# - S-STR-03: description field exists
# - S-STR-04: references/ directory not empty (if exists)
# - S-STR-05: name length 1-64 characters
# - S-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
# - S-STR-07: description length 1-1024 characters
# - S-STR-08: All links point to existing files
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Skill Structure ==="
echo ""
echo "This test validates structure for all skills."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

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
print_section_header "Test: Skill Structure (S-STR-01 to S-STR-07)"

for skill in $ALL_SKILLS; do
    skill_file="$SKILLS_DIR/skills/$skill/SKILL.md"
    
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

for skill_path in "$SKILLS_DIR/skills"/*/SKILL.md; do
    if [ -f "$skill_path" ]; then
        if check_file_links "$skill_path" "skill"; then
            ((link_pass++)) || true
        else
            ((link_fail++)) || true
        fi
    fi
done

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
echo ""

if [ $((structure_fail + link_fail)) -gt 0 ]; then
    print_status_failed
    echo ""
    echo "Please fix the failed structure checks."
    exit 1
else
    print_status_passed
    exit 0
fi