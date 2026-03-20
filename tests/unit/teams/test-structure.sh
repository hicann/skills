#!/usr/bin/env bash
# =============================================================================
# Test: Team Structure
# =============================================================================
# Validates structure correctness for all teams.
# Rules tested:
# - T-STR-01: YAML Front Matter format (---wrapped)
# - T-STR-02: description field exists
# - T-STR-03: mode field exists and is "primary"
# - T-STR-04: skills field exists
# - T-STR-05: All skill dependencies exist
# - T-STR-06: description length 1-1024 characters
# - T-STR-07: references/ directory not empty (if exists)
# - T-STR-08: All links point to existing files
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Team Structure ==="
echo ""
echo "This test validates structure for all teams."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

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
print_section_header "Test: Team Structure (T-STR-01 to T-STR-07)"

for team in $ALL_TEAMS; do
    team_file="$SKILLS_DIR/teams/$team/AGENTS.md"
    
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
print_section_header "Test: Link Validity (T-STR-08)"

for team_path in "$SKILLS_DIR/teams"/*/AGENTS.md; do
    if [ -f "$team_path" ]; then
        if check_file_links "$team_path" "team"; then
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
echo -e " ${BOLD}Team Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total teams: $total_teams"
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