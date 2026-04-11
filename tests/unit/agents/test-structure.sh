#!/usr/bin/env bash
# =============================================================================
# Test: Agent Structure
# =============================================================================
# Validates structure correctness for all agents.
# Rules tested:
# - A-STR-01: YAML Front Matter format
# - A-STR-02: name/description/mode fields exist
# - A-STR-03: mode is primary or subagent
# - A-STR-04: All skill dependencies exist
# - A-STR-05: name length 1-64 characters
# - A-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
# - A-STR-07: description length 1-1024 characters
# - A-STR-08: All links point to existing files
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Structure ==="
echo ""
echo "This test validates structure for all agents."
echo "Run time: ~15 seconds (no CLI needed)"
echo ""

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
print_section_header "Test: Agent Structure (A-STR-01 to A-STR-07)"

for agent in $ALL_AGENTS; do
    agent_file=$(find_agent_file "$agent")
    
    if [ ! -f "$agent_file" ]; then
        print_skip "$agent: AGENT.md not found"
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
# Summary
# ============================================
echo "========================================"
echo -e " ${BOLD}Agent Structure Test Summary${NC}"
echo "========================================"
echo ""
echo "  Total agents: $total_agents"
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