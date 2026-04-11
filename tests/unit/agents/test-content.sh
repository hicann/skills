#!/usr/bin/env bash
# =============================================================================
# Test: Agent Content
# =============================================================================
# Validates content quality for all agents.
# Rules tested:
# - A-CON-01: name matches directory name
# - A-CON-02: description contains trigger keywords
# - A-CON-03: naming follows prefix convention
# - A-CON-04: has core responsibilities section
# - A-CON-05: has responsibility boundary (recommended)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Agent Content ==="
echo ""
echo "This test validates content quality for all agents."
echo "Run time: ~10 seconds (no CLI needed)"
echo ""

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
print_section_header "Test: Agent Content (A-CON-01 to A-CON-05)"

for agent in $ALL_AGENTS; do
    agent_file=$(find_agent_file "$agent")
    
    if [ ! -f "$agent_file" ]; then
        print_skip "$agent: AGENT.md not found"
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