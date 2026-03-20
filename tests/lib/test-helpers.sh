#!/usr/bin/env bash
# =============================================================================
# Helper functions for CANN Skills tests
# =============================================================================
# Supports: Claude Code + OpenCode dual platform
# Features: Test isolation, CI/CD ready, JSON output
# =============================================================================

set -euo pipefail

# Get the directories
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$LIB_DIR/.." && pwd)"
SKILLS_DIR="$(cd "$LIB_DIR/../.." && pwd)"

# =============================================================================
# Color Output
# =============================================================================
# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if colors should be enabled (respects NO_COLOR env var and non-TTY)
# Supports FORCE_COLOR=1 to force enable colors in CI/non-TTY environments
setup_colors() {
    if [ -n "${NO_COLOR:-}" ]; then
        disable_colors
    elif [ -n "${FORCE_COLOR:-}" ] || [ "${FORCE_COLOR:-}" = "1" ]; then
        enable_colors
    elif [ -t 1 ]; then
        enable_colors
    else
        disable_colors
    fi
}

enable_colors() {
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
}

disable_colors() {
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
}

# Print colored status - these are the main functions to use
print_pass() {
    echo -e "  ${GREEN}[PASS]${NC} $*"
}

print_fail() {
    echo -e "  ${RED}[FAIL]${NC} $*"
}

print_skip() {
    echo -e "  ${YELLOW}[SKIP]${NC} $*"
}

print_info() {
    echo -e "  ${BLUE}[INFO]${NC} $*"
}

print_warn() {
    echo -e "  ${YELLOW}[WARN]${NC} $*"
}

print_error() {
    echo -e "  ${RED}[ERROR]${NC} $*"
}

print_section_header() {
    local name="$1"
    echo ""
    echo -e "${BOLD}${CYAN}=== $name ===${NC}"
    echo ""
}

# Print colored summary status
print_status_passed() {
    echo -e "${GREEN}${BOLD}STATUS: PASSED${NC}"
}

print_status_failed() {
    echo -e "${RED}${BOLD}STATUS: FAILED${NC}"
}

# Initialize colors at source time
setup_colors

# =============================================================================
# Configuration
# =============================================================================

# Default configuration
DEFAULT_TIMEOUT=60
DEFAULT_MAX_TURNS=3
DEFAULT_PLATFORM="opencode"  # claude, opencode, or all

# Test results tracking
declare -g TEST_PASSED=0
declare -g TEST_FAILED=0
declare -g TEST_SKIPPED=0
declare -g TEST_START_TIME=0
declare -g TEST_OUTPUT_DIR=""

# =============================================================================
# Platform Detection
# =============================================================================

# Detect available platforms
detect_platforms() {
    local platforms=""
    command -v claude &> /dev/null && platforms="$platforms claude"
    command -v opencode &> /dev/null && platforms="$platforms opencode"
    echo "$platforms"
}

# Check if a specific platform is available
# Usage: is_platform_available "claude"
is_platform_available() {
    local platform="$1"
    case "$platform" in
        claude)  command -v claude &> /dev/null ;;
        opencode) command -v opencode &> /dev/null ;;
        *) return 1 ;;
    esac
}

# Get platform version
# Usage: get_platform_version "claude"
get_platform_version() {
    local platform="$1"
    case "$platform" in
        claude)  claude --version 2>/dev/null || echo "unknown" ;;
        opencode) opencode --version 2>/dev/null || echo "unknown" ;;
        *) echo "unknown" ;;
    esac
}

# =============================================================================
# Test Project Management
# =============================================================================

# Create a temporary test project directory
# Usage: test_project=$(create_test_project [prefix])
create_test_project() {
    local prefix="${1:-cann-test}"
    local test_dir=$(mktemp -d -t "${prefix}.XXXXXX")
    echo "$test_dir"
}

# Cleanup test project
# Usage: cleanup_test_project "$test_dir"
cleanup_test_project() {
    local test_dir="$1"
    if [ -d "$test_dir" ]; then
        rm -rf "$test_dir"
    fi
}

# =============================================================================
# Platform-specific Runners
# =============================================================================

# Run Claude Code with a prompt and capture output
# Usage: run_claude "prompt text" [timeout_seconds] [allowed_tools]
run_claude() {
    local prompt="$1"
    local timeout="${2:-$DEFAULT_TIMEOUT}"
    local allowed_tools="${3:-}"
    local output_file=$(mktemp)

    # Build command
    local cmd="claude -p \"$prompt\""
    if [ -n "$allowed_tools" ]; then
        cmd="$cmd --allowed-tools=$allowed_tools"
    fi

    # Run Claude in headless mode with timeout
    if timeout "$timeout" bash -c "$cmd" > "$output_file" 2>&1; then
        cat "$output_file"
        rm -f "$output_file"
        return 0
    else
        local exit_code=$?
        cat "$output_file" >&2
        rm -f "$output_file"
        return $exit_code
    fi
}

# Run OpenCode with a prompt and capture output
# Usage: run_opencode "prompt text" [timeout_seconds]
run_opencode() {
    local prompt="$1"
    local timeout="${2:-$DEFAULT_TIMEOUT}"
    local output_file=$(mktemp)

    # Run OpenCode in headless mode
    if timeout "$timeout" opencode -p "$prompt" > "$output_file" 2>&1; then
        cat "$output_file"
        rm -f "$output_file"
        return 0
    else
        local exit_code=$?
        cat "$output_file" >&2
        rm -f "$output_file"
        return $exit_code
    fi
}

# Universal runner - auto-detect platform
# Usage: run_ai "prompt text" [timeout_seconds] [platform]
run_ai() {
    local prompt="$1"
    local timeout="${2:-$DEFAULT_TIMEOUT}"
    local platform="${3:-$DEFAULT_PLATFORM}"

    case "$platform" in
        claude)  run_claude "$prompt" "$timeout" ;;
        opencode) run_opencode "$prompt" "$timeout" ;;
        *)
            echo "[ERROR] Unknown platform: $platform"
            return 1
            ;;
    esac
}

# =============================================================================
# Assertions
# =============================================================================

# Check if output contains a pattern (case-insensitive)
# Usage: assert_contains "output" "pattern" "test name"
assert_contains() {
    local output="$1"
    local pattern="$2"
    local test_name="${3:-test}"

    if echo "$output" | grep -qiE "$pattern"; then
        print_pass "$test_name"
        return 0
    else
        print_fail "$test_name"
        echo -e "  ${YELLOW}Expected to find:${NC} $pattern"
        echo "  In output:"
        echo "$output" | sed 's/^/    /'
        return 1
    fi
}

# Check if output does NOT contain a pattern
# Usage: assert_not_contains "output" "pattern" "test name"
assert_not_contains() {
    local output="$1"
    local pattern="$2"
    local test_name="${3:-test}"

    if echo "$output" | grep -qiE "$pattern"; then
        print_fail "$test_name"
        echo -e "  ${YELLOW}Did not expect to find:${NC} $pattern"
        echo "  In output:"
        echo "$output" | sed 's/^/    /'
        return 1
    else
        print_pass "$test_name"
        return 0
    fi
}

# Check if output matches a count
# Usage: assert_count "output" "pattern" expected_count "test name"
assert_count() {
    local output="$1"
    local pattern="$2"
    local expected="$3"
    local test_name="${4:-test}"

    local actual=$(echo "$output" | grep -ciE "$pattern" || echo "0")

    if [ "$actual" -eq "$expected" ]; then
        print_pass "$test_name (found $actual instances)"
        return 0
    else
        print_fail "$test_name"
        echo -e "  ${YELLOW}Expected $expected instances of:${NC} $pattern"
        echo "  Found $actual instances"
        echo "  In output:"
        echo "$output" | sed 's/^/    /'
        return 1
    fi
}

# Check if pattern A appears before pattern B
# Usage: assert_order "output" "pattern_a" "pattern_b" "test name"
assert_order() {
    local output="$1"
    local pattern_a="$2"
    local pattern_b="$3"
    local test_name="${4:-test}"

    local line_a=$(echo "$output" | grep -inE "$pattern_a" | head -1 | cut -d: -f1)
    local line_b=$(echo "$output" | grep -inE "$pattern_b" | head -1 | cut -d: -f1)

    if [ -z "$line_a" ]; then
        print_fail "$test_name: pattern A not found: $pattern_a"
        return 1
    fi

    if [ -z "$line_b" ]; then
        print_fail "$test_name: pattern B not found: $pattern_b"
        return 1
    fi

    if [ "$line_a" -lt "$line_b" ]; then
        print_pass "$test_name (A at line $line_a, B at line $line_b)"
        return 0
    elif [ "$line_a" -eq "$line_b" ]; then
        print_pass "$test_name (both on line $line_a)"
        return 0
    else
        print_fail "$test_name"
        echo -e "  ${YELLOW}Expected '$pattern_a' before '$pattern_b'${NC}"
        echo "  But found A at line $line_a, B at line $line_b"
        return 1
    fi
}

# Assert file exists
# Usage: assert_file_exists "/path/to/file" "test name"
assert_file_exists() {
    local file="$1"
    local test_name="${2:-file exists}"

    if [ -f "$file" ]; then
        print_pass "$test_name"
        return 0
    else
        print_fail "$test_name"
        echo "  File not found: $file"
        return 1
    fi
}

# =============================================================================
# File Link Validation
# =============================================================================

# Check link validity in a file (for SKILL.md and AGENT.md)
# Usage: check_file_links "/path/to/file.md" "skill|agent"
# Returns: 0 if all links valid, 1 if broken links found
check_file_links() {
    local file="$1"
    local file_type="$2"
    local file_dir="$(dirname "$file")"
    local item_name=$(basename "$file_dir")
    local broken_links=()

    while IFS= read -r line; do
        # Pattern: [text](references/...)
        if echo "$line" | grep -qE '\]\(references/'; then
            local link=$(echo "$line" | sed -n 's/.*(\(references\/[^)]*\)).*/\1/p' | head -1 | cut -d'#' -f1)
            if [ -n "$link" ] && [ ! -e "$file_dir/$link" ]; then
                broken_links+=("$link")
            fi
        fi

        # Pattern: {file:./references/...}
        if echo "$line" | grep -qE '\{file:'; then
            local link=$(echo "$line" | sed -n 's/.*{file:\([^}]*\)}.*/\1/p' | head -1 | cut -d'#' -f1)
            link="${link#./}"
            if [ -n "$link" ] && [ ! -e "$file_dir/$link" ]; then
                broken_links+=("$link")
            fi
        fi

        # Pattern: [text](./path)
        if echo "$line" | grep -qE '\]\(\./'; then
            local link=$(echo "$line" | sed -n 's/.*(\./\([^)]*\)).*/\1/p' | head -1 | cut -d'#' -f1)
            if [ -n "$link" ] && [ ! -e "$file_dir/$link" ]; then
                broken_links+=("$link")
            fi
        fi
    done < "$file"

    if [ ${#broken_links[@]} -gt 0 ]; then
        print_fail "$file_type/$item_name: Broken links:"
        echo -e "    ${YELLOW}${broken_links[*]}${NC}"
        return 1
    else
        print_pass "$file_type/$item_name: All links valid"
        return 0
    fi
}

# =============================================================================
# YAML Data Extraction
# =============================================================================

# Extract skills list from agent YAML front matter
# Usage: extract_agent_skills "/path/to/AGENT.md"
# Returns: space-separated list of skill names
extract_agent_skills() {
    local agent_file="$1"
    local skills=()
    local in_skills=false

    while IFS= read -r line; do
        if [[ "$line" == "skills:" ]]; then
            in_skills=true
            continue
        fi

        if $in_skills; then
            if [[ "$line" =~ ^[[:space:]]+-[[:space:]]+(.+)$ ]]; then
                skills+=("${BASH_REMATCH[1]}")
            elif [[ ! "$line" =~ ^[[:space:]] ]] && [[ -n "$line" ]] && [[ "$line" != "---" ]]; then
                break
            fi
        fi
    done < "$agent_file"

    echo "${skills[@]}"
}

# =============================================================================
# Behavior Test Helpers
# =============================================================================

# Run a behavior test with Claude CLI
# Usage: run_behavior_test "test_name" "prompt" "expected_pattern" [timeout]
# Globals: pass_count, fail_count, skip_count should be defined before calling
run_behavior_test() {
    local name="$1"
    local prompt="$2"
    local expected="$3"
    local timeout="${4:-25}"
    
    echo "Testing: $name"
    
    if output=$(timeout "$timeout" claude -p "$prompt 用 1 行回答。" 2>&1); then
        if echo "$output" | grep -qiE "$expected"; then
            print_pass "Correct response"
            pass_count=$((pass_count + 1))
        else
            print_fail "Incorrect response"
            echo -e "  ${YELLOW}Expected:${NC} $expected"
            echo "  Output: ${output:0:80}..."
            fail_count=$((fail_count + 1))
        fi
    else
        print_skip "Claude CLI timed out"
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# =============================================================================
# Premature Action Analysis
# =============================================================================

# Analyze premature actions in a session file
# Usage: analyze_premature_actions "$session_file" "skill|agent" "target_name"
analyze_premature_actions() {
    local session_file="$1"
    local target_type="$2"  # "skill" or "agent"
    local target_name="$3"
    
    if [ ! -f "$session_file" ]; then
        echo "  [SKIP] Session file not found"
        return 0
    fi
    
    local passed=true
    local tool_invoked=false
    local first_tool_line=""
    
    # Find the first tool invocation (Skill or Task)
    local search_pattern='"name":"Skill"'
    [ "$target_type" = "agent" ] && search_pattern='"name":"Task"'
    
    if grep -q "$search_pattern" "$session_file" 2>/dev/null; then
        tool_invoked=true
        first_tool_line=$(grep -n "$search_pattern" "$session_file" 2>/dev/null | head -1 | cut -d: -f1)
    fi
    
    # Check if the target was invoked
    if $tool_invoked; then
        local target_pattern
        if [ "$target_type" = "skill" ]; then
            target_pattern='"skill":"([^"]*:)?'"${target_name}"'"'
        else
            target_pattern="\"subagent_type\":\"[^\"]*${target_name}[^\"]*\""
        fi
        
        if grep -qE "$target_pattern" "$session_file" 2>/dev/null; then
            print_pass "Target $target_type '$target_name' was invoked"
        else
            print_info "Other ${target_type}s were invoked instead of '$target_name'"
            if [ "$target_type" = "skill" ]; then
                local triggered=$(grep -o '"skill":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/^/    - /')
            else
                local triggered=$(grep -o '"subagent_type":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/^/    - /')
            fi
            if [ -n "$triggered" ]; then
                print_info "Triggered ${target_type}s:"
                echo "$triggered"
            fi
        fi
    else
        if [ "$target_type" = "skill" ]; then
            print_warn "No Skill tool was invoked in this session"
        else
            print_info "No explicit Task tool invocation (agent may be triggered implicitly)"
        fi
    fi
    
    # Check for premature Write/Edit operations
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        local premature_tools=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep '"type":"tool_use"' 2>/dev/null | \
            grep -v '"name":"Skill"' 2>/dev/null | \
            grep -v '"name":"Task"' 2>/dev/null | \
            grep -v '"name":"TodoWrite"' 2>/dev/null | \
            grep -v '"name":"Read"' 2>/dev/null | \
            grep -v '"name":"Glob"' 2>/dev/null | \
            grep -v '"name":"Grep"' 2>/dev/null | \
            grep -E '"name":"(Write|Edit|Bash)"' 2>/dev/null || true)
        
        if [ -n "$premature_tools" ]; then
            print_fail "Premature actions detected BEFORE $target_type invocation:"
            echo "$premature_tools" | head -5 | sed 's/^/    /'
            passed=false
        else
            print_pass "No premature Write/Edit/Bash actions before $target_type"
        fi
    fi
    
    # Check for TodoWrite usage (acceptable before tool)
    local todo_before=0
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        todo_before=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep -c '"name":"TodoWrite"' 2>/dev/null || echo "0")
    fi
    
    if [ "$todo_before" -gt 0 ]; then
        print_info "TodoWrite used $todo_before time(s) before $target_type (acceptable for planning)"
    fi
    
    # Check for Read usage before tool (acceptable for context)
    local read_before=0
    if $tool_invoked && [ -n "$first_tool_line" ]; then
        read_before=$(head -n "$first_tool_line" "$session_file" 2>/dev/null | \
            grep -c '"name":"Read"' 2>/dev/null || echo "0")
    fi
    
    if [ "$read_before" -gt 0 ]; then
        print_info "Read used $read_before time(s) before $target_type (acceptable for context)"
    fi
    
    # Summary
    if $passed; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Skill & Agent Queries
# =============================================================================

# Get list of all skills
get_all_skills() {
    find "$SKILLS_DIR/skills" -maxdepth 2 -name "SKILL.md" -exec dirname {} \; 2>/dev/null | xargs -I{} basename {} | sort
}

# Get list of all agents
get_all_agents() {
    find "$SKILLS_DIR/agents" -maxdepth 2 -name "AGENT.md" -exec dirname {} \; 2>/dev/null | xargs -I{} basename {} | sort
}

# Get list of all teams
get_all_teams() {
    find "$SKILLS_DIR/teams" -maxdepth 2 -name "AGENTS.md" -exec dirname {} \; 2>/dev/null | xargs -I{} basename {} | sort
}

# =============================================================================
# Structure & Content Validation Functions (Auto-scan)
# =============================================================================

# Skill description keywords (trigger keywords)
SKILL_KEYWORDS="Ascend|算子|Kernel|Tiling|调试|debug|测试|test|性能|perf|精度|precision|NPU|开发|API|ACLNN|运行时|runtime"

# Required sections for agents (regex patterns)
AGENT_REQUIRED_SECTIONS="核心职责|概述|核心原则|核心工作流程|Core Responsibilities|Overview"
AGENT_RECOMMENDED_SECTIONS="工作流程|场景|Workflow|Scene"

# Validate skill STRUCTURE
# Rules: S-STR-01 to S-STR-07
# Returns: 0 if valid, 1 if errors found
validate_skill_structure() {
    local skill_file="$1"
    local skill_name=$(basename $(dirname "$skill_file"))
    local errors=()
    
    # S-STR-01: YAML format
    if ! head -1 "$skill_file" | grep -q "^---$"; then
        errors+=("S-STR-01: Missing opening ---")
    fi
    if ! head -20 "$skill_file" | grep -q "^---$"; then
        errors+=("S-STR-01: Missing closing ---")
    fi
    
    # S-STR-02: name field exists
    if ! grep -q "^name:" "$skill_file"; then
        errors+=("S-STR-02: Missing 'name' field")
    fi
    
    # S-STR-03: description field exists
    if ! grep -q "^description:" "$skill_file"; then
        errors+=("S-STR-03: Missing 'description' field")
    fi
    
    # S-STR-04: references directory not empty (if exists)
    local ref_dir="$(dirname "$skill_file")/references"
    if [ -d "$ref_dir" ]; then
        local ref_count=$(find "$ref_dir" -name "*.md" -type f 2>/dev/null | wc -l)
        if [ "$ref_count" -eq 0 ]; then
            errors+=("S-STR-04: Empty references directory")
        fi
    fi
    
    # S-STR-05: name length 1-64 characters
    local yaml_name=$(grep "^name:" "$skill_file" | head -1 | sed 's/^name:[[:space:]]*//' | tr -d '[:space:]')
    if [ -n "$yaml_name" ]; then
        local name_len=${#yaml_name}
        if [ "$name_len" -lt 1 ] || [ "$name_len" -gt 64 ]; then
            errors+=("S-STR-05: name length must be 1-64 chars (got $name_len)")
        fi
    fi
    
    # S-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
    if [ -n "$yaml_name" ] && ! echo "$yaml_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("S-STR-06: name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$yaml_name')")
    fi
    
    # S-STR-07: description length 1-1024 characters
    local description=$(grep "^description:" "$skill_file" | head -1 | sed 's/^description:[[:space:]]*//')
    if [ -n "$description" ]; then
        local desc_len=${#description}
        if [ "$desc_len" -lt 1 ] || [ "$desc_len" -gt 1024 ]; then
            errors+=("S-STR-07: description length must be 1-1024 chars (got $desc_len)")
        fi
    fi
    
    # Note: S-STR-08 (Link validity) is checked separately in test-structure.sh
    
    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$skill_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$skill_name: Structure valid"
        return 0
    fi
}

# Validate skill CONTENT
# Rules: S-CON-01 to S-CON-04
# Returns: 0 if valid, 1 if errors found
validate_skill_content() {
    local skill_file="$1"
    local skill_name=$(basename $(dirname "$skill_file"))
    local errors=()
    local warnings=()
    
    # S-CON-01: name matches directory name
    local yaml_name=$(grep "^name:" "$skill_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$yaml_name" ] && [ "$yaml_name" != "$skill_name" ]; then
        errors+=("S-CON-01: name '$yaml_name' != directory '$skill_name'")
    fi
    
    # S-CON-02: description has trigger keywords
    local description=$(grep "^description:" "$skill_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$SKILL_KEYWORDS"; then
        errors+=("S-CON-02: Description lacks trigger keywords")
    fi
    
    # S-CON-03: description has trigger conditions (recommended)
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "触发|Trigger|使用|场景|条件"; then
        warnings+=("S-CON-03: Missing trigger conditions in description")
    fi
    
    # S-CON-04: naming prefix convention
    if ! echo "$skill_name" | grep -qE "^(cann-|ascendc-|[a-z]+-)"; then
        errors+=("S-CON-04: Naming must have prefix (cann-, ascendc-, etc.)")
    fi
    
    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$skill_name: ${#errors[@]} error(s), ${#warnings[@]} warning(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        for warn in "${warnings[@]}"; do echo -e "    ${YELLOW}[WARN]${NC} $skill_name: $warn"; done
        return 1
    elif [ ${#warnings[@]} -gt 0 ]; then
        print_pass "$skill_name: Valid (${#warnings[@]} warning(s))"
        for warn in "${warnings[@]}"; do echo -e "    ${YELLOW}[WARN]${NC} $skill_name: $warn"; done
        return 0
    else
        print_pass "$skill_name: Content valid"
        return 0
    fi
}

# Validate agent STRUCTURE
# Rules: A-STR-01 to A-STR-07
# Returns: 0 if valid, 1 if errors found
validate_agent_structure() {
    local agent_file="$1"
    local agent_name=$(basename $(dirname "$agent_file"))
    local errors=()
    
    # A-STR-01: YAML format
    if ! head -1 "$agent_file" | grep -q "^---$"; then
        errors+=("A-STR-01: Missing opening ---")
    fi
    if ! head -20 "$agent_file" | grep -q "^---$"; then
        errors+=("A-STR-01: Missing closing ---")
    fi
    
    # A-STR-02: name/description/mode fields exist
    for field in name description mode; do
        if ! grep -q "^$field:" "$agent_file"; then
            errors+=("A-STR-02: Missing '$field' field")
        fi
    done
    
    # A-STR-03: mode is primary or subagent
    local mode=$(grep "^mode:" "$agent_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$mode" ] && [[ "$mode" != "primary" && "$mode" != "subagent" ]]; then
        errors+=("A-STR-03: Invalid mode '$mode' (must be: primary or subagent)")
    fi
    
    # A-STR-04: skills dependencies exist
    local skills=$(extract_agent_skills "$agent_file")
    for skill in $skills; do
        local skill_file="$SKILLS_DIR/skills/$skill/SKILL.md"
        if [ ! -f "$skill_file" ]; then
            errors+=("A-STR-04: Missing skill dependency: $skill")
        fi
    done
    
    # A-STR-05: name length 1-64 characters
    local yaml_name=$(grep "^name:" "$agent_file" | head -1 | sed 's/^name:[[:space:]]*//' | tr -d '[:space:]')
    if [ -n "$yaml_name" ]; then
        local name_len=${#yaml_name}
        if [ "$name_len" -lt 1 ] || [ "$name_len" -gt 64 ]; then
            errors+=("A-STR-05: name length must be 1-64 chars (got $name_len)")
        fi
    fi
    
    # A-STR-06: name format ^[a-z0-9]+(-[a-z0-9]+)*$
    if [ -n "$yaml_name" ] && ! echo "$yaml_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("A-STR-06: name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$yaml_name')")
    fi
    
    # A-STR-07: description length 1-1024 characters
    local description=$(grep "^description:" "$agent_file" | head -1 | sed 's/^description:[[:space:]]*//')
    if [ -n "$description" ]; then
        local desc_len=${#description}
        if [ "$desc_len" -lt 1 ] || [ "$desc_len" -gt 1024 ]; then
            errors+=("A-STR-07: description length must be 1-1024 chars (got $desc_len)")
        fi
    fi
    
    # Note: A-STR-08 (Link validity) is checked separately in test-structure.sh
    
    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$agent_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$agent_name: Structure valid"
        return 0
    fi
}

# Validate agent CONTENT
# Rules: A-CON-01 to A-CON-05
# Returns: 0 if valid, 1 if errors found
validate_agent_content() {
    local agent_file="$1"
    local agent_name=$(basename $(dirname "$agent_file"))
    local errors=()
    local warnings=()
    
    # A-CON-01: name matches directory name
    local yaml_name=$(grep "^name:" "$agent_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$yaml_name" ] && [ "$yaml_name" != "$agent_name" ]; then
        errors+=("A-CON-01: name '$yaml_name' != directory '$agent_name'")
    fi
    
    # A-CON-02: description has trigger keywords
    local description=$(grep "^description:" "$agent_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$SKILL_KEYWORDS"; then
        errors+=("A-CON-02: Description lacks trigger keywords")
    fi
    
    # A-CON-03: naming prefix convention
    if ! echo "$agent_name" | grep -qE "^(cann-|ascendc-|[a-z]+-)"; then
        errors+=("A-CON-03: Naming must have prefix (cann-, ascendc-, etc.)")
    fi
    
    # A-CON-04: core responsibilities section (required)
    if ! grep -qE "^#+ *($AGENT_REQUIRED_SECTIONS)" "$agent_file"; then
        errors+=("A-CON-04: Missing core responsibilities section")
    fi
    
    # A-CON-05: removed per user request (was: responsibility boundary)
    
    # A-CON-07: workflow section (recommended, removed per user request)
    
    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$agent_name: ${#errors[@]} error(s), ${#warnings[@]} warning(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        for warn in "${warnings[@]}"; do echo -e "    ${YELLOW}[WARN]${NC} $agent_name: $warn"; done
        return 1
    elif [ ${#warnings[@]} -gt 0 ]; then
        print_pass "$agent_name: Valid (${#warnings[@]} warning(s))"
        for warn in "${warnings[@]}"; do echo -e "    ${YELLOW}[WARN]${NC} $agent_name: $warn"; done
        return 0
    else
        print_pass "$agent_name: Content valid"
        return 0
    fi
}

# =============================================================================
# Team Structure & Content Validation Functions
# =============================================================================

# Team description keywords (trigger keywords)
TEAM_KEYWORDS="团队|Team|协同|编排|流程|开发|Agent|多Agent"

# Extract skills list from team YAML front matter
# Usage: extract_team_skills "/path/to/AGENTS.md"
# Returns: space-separated list of skill names
extract_team_skills() {
    local team_file="$1"
    local skills=()
    local in_skills=false

    while IFS= read -r line; do
        if [[ "$line" == "skills:" ]]; then
            in_skills=true
            continue
        fi

        if $in_skills; then
            if [[ "$line" =~ ^[[:space:]]+-[[:space:]]+(.+)$ ]]; then
                skills+=("${BASH_REMATCH[1]}")
            elif [[ ! "$line" =~ ^[[:space:]] ]] && [[ -n "$line" ]] && [[ "$line" != "---" ]]; then
                break
            fi
        fi
    done < "$team_file"

    echo "${skills[@]}"
}

# Validate team STRUCTURE
# Rules: T-STR-01 to T-STR-07
# Returns: 0 if valid, 1 if errors found
validate_team_structure() {
    local team_file="$1"
    local team_name=$(basename $(dirname "$team_file"))
    local errors=()

    # T-STR-01: YAML format
    if ! head -1 "$team_file" | grep -q "^---$"; then
        errors+=("T-STR-01: Missing opening ---")
    fi
    if ! head -20 "$team_file" | grep -q "^---$"; then
        errors+=("T-STR-01: Missing closing ---")
    fi

    # T-STR-02: description field exists
    if ! grep -q "^description:" "$team_file"; then
        errors+=("T-STR-02: Missing 'description' field")
    fi

    # T-STR-03: mode field exists and is "primary"
    local mode=$(grep "^mode:" "$team_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -z "$mode" ]; then
        errors+=("T-STR-03: Missing 'mode' field")
    elif [ "$mode" != "primary" ]; then
        errors+=("T-STR-03: Invalid mode '$mode' (must be: primary)")
    fi

    # T-STR-04: skills field exists
    if ! grep -q "^skills:" "$team_file"; then
        errors+=("T-STR-04: Missing 'skills' field")
    fi

    # T-STR-05: skills dependencies exist
    local skills=$(extract_team_skills "$team_file")
    for skill in $skills; do
        local skill_file="$SKILLS_DIR/skills/$skill/SKILL.md"
        if [ ! -f "$skill_file" ]; then
            errors+=("T-STR-05: Missing skill dependency: $skill")
        fi
    done

    # T-STR-06: description length 1-1024 characters
    local description=$(grep "^description:" "$team_file" | head -1 | sed 's/^description:[[:space:]]*//')
    if [ -n "$description" ]; then
        local desc_len=${#description}
        if [ "$desc_len" -lt 1 ] || [ "$desc_len" -gt 1024 ]; then
            errors+=("T-STR-06: description length must be 1-1024 chars (got $desc_len)")
        fi
    fi

    # T-STR-07: references directory not empty (if exists)
    local ref_dir="$(dirname "$team_file")/references"
    if [ -d "$ref_dir" ]; then
        local ref_count=$(find "$ref_dir" -name "*.md" -type f 2>/dev/null | wc -l)
        if [ "$ref_count" -eq 0 ]; then
            errors+=("T-STR-07: Empty references directory")
        fi
    fi

    # Note: T-STR-08 (Link validity) is checked separately in test-structure.sh

    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$team_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$team_name: Structure valid"
        return 0
    fi
}

# Validate team CONTENT
# Rules: T-CON-01 to T-CON-05
# Returns: 0 if valid, 1 if errors found
validate_team_content() {
    local team_file="$1"
    local team_name=$(basename $(dirname "$team_file"))
    local errors=()
    local warnings=()
    local infos=()

    # T-CON-01: directory naming format
    if ! echo "$team_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("T-CON-01: Directory name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$team_name')")
    fi

    # T-CON-02: description has trigger keywords
    local description=$(grep "^description:" "$team_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$TEAM_KEYWORDS"; then
        errors+=("T-CON-02: Description lacks trigger keywords")
    fi

    # T-CON-03: core principles section (required)
    if ! grep -qE "^#+ *(核心原则|Core Principles)" "$team_file"; then
        errors+=("T-CON-03: Missing core principles section")
    fi

    # T-CON-04: init.sh exists (optional)
    local init_script="$(dirname "$team_file")/init.sh"
    if [ ! -f "$init_script" ]; then
        infos+=("T-CON-04: No init.sh script found")
    fi

    # T-CON-05: quickstart.md exists (optional)
    local quickstart="$(dirname "$team_file")/quickstart.md"
    if [ ! -f "$quickstart" ]; then
        infos+=("T-CON-05: No quickstart.md found")
    fi

    # Output
    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$team_name: ${#errors[@]} error(s), ${#warnings[@]} warning(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        for warn in "${warnings[@]}"; do print_warn "$warn"; done
        for info in "${infos[@]}"; do print_info "$info"; done
        return 1
    elif [ ${#warnings[@]} -gt 0 ]; then
        print_pass "$team_name: Valid (${#warnings[@]} warning(s))"
        for warn in "${warnings[@]}"; do print_warn "$warn"; done
        for info in "${infos[@]}"; do print_info "$info"; done
        return 0
    else
        print_pass "$team_name: Content valid"
        for info in "${infos[@]}"; do print_info "$info"; done
        return 0
    fi
}

# =============================================================================
# Session Analysis
# =============================================================================

# Find the most recent session file
# Usage: session_file=$(find_recent_session [minutes_old])
find_recent_session() {
    local minutes_old="${1:-30}"
    local session_dir="$HOME/.claude/projects"

    # Find the most recent session file
    find "$session_dir" -name "*.jsonl" -type f -mmin -"$minutes_old" 2>/dev/null | sort -r | head -1
}

# Check if a skill was invoked in a session
# Usage: verify_skill_invoked "$session_file" "skill-name"
verify_skill_invoked() {
    local session_file="$1"
    local skill_name="$2"

    if [ ! -f "$session_file" ]; then
        print_fail "Session file not found: $session_file"
        return 1
    fi

    # Look for Skill tool invocation with the skill name
    # Match both "skill":"skillname" and "skill":"namespace:skillname"
    local skill_pattern='"skill":"([^"]*:)?'"${skill_name}"'"'

    if grep -q '"name":"Skill"' "$session_file" && grep -qE "$skill_pattern" "$session_file"; then
        print_pass "Skill '$skill_name' was invoked"
        return 0
    else
        print_fail "Skill '$skill_name' was NOT invoked"
        return 1
    fi
}

# Check if an agent was dispatched in a session
# Usage: verify_agent_dispatched "$session_file" "agent-name"
verify_agent_dispatched() {
    local session_file="$1"
    local agent_name="$2"

    if [ ! -f "$session_file" ]; then
        print_fail "Session file not found: $session_file"
        return 1
    fi

    # Look for Agent tool invocation or Task with agent subagent_type
    if grep -qE '"subagent_type":"'"$agent_name"'"|"name":"'"$agent_name"'"' "$session_file"; then
        print_pass "Agent '$agent_name' was dispatched"
        return 0
    else
        print_fail "Agent '$agent_name' was NOT dispatched"
        return 1
    fi
}

# Count tool invocations in a session
# Usage: count=$(count_tool_invocations "$session_file" "ToolName")
count_tool_invocations() {
    local session_file="$1"
    local tool_name="$2"

    if [ ! -f "$session_file" ]; then
        echo "0"
        return
    fi

    local count=$(grep -c "\"name\":\"$tool_name\"" "$session_file" 2>/dev/null || echo "0")
    echo "$count"
}

# Check for premature action (tools invoked before skill)
# Usage: check_premature_action "$session_file" "skill-name"
check_premature_action() {
    local session_file="$1"
    local skill_name="$2"

    if [ ! -f "$session_file" ]; then
        print_skip "Session file not found"
        return 0
    fi

    # Find the line number of the first Skill invocation
    local first_skill_line=$(grep -n '"name":"Skill"' "$session_file" 2>/dev/null | head -1 | cut -d: -f1 || true)

    if [ -z "$first_skill_line" ]; then
        print_warn "No Skill invocation found"
        return 0
    fi

    # Check for tool invocations before the Skill invocation
    local premature_tools=$(head -n "$first_skill_line" "$session_file" 2>/dev/null | \
        grep '"type":"tool_use"' 2>/dev/null | \
        grep -v '"name":"Skill"' 2>/dev/null | \
        grep -v '"name":"TodoWrite"' 2>/dev/null | \
        grep -v '"name":"TaskOutput"' 2>/dev/null || true)

    if [ -n "$premature_tools" ]; then
        print_warn "Tools invoked BEFORE Skill '$skill_name':"
        echo "$premature_tools" | head -3 | sed 's/^/    /' || true
        return 1
    else
        print_pass "No premature tool invocations detected"
        return 0
    fi
}

# Get list of triggered skills from session
# Usage: get_triggered_skills "$session_file"
get_triggered_skills() {
    local session_file="$1"
    if [ ! -f "$session_file" ]; then
        return
    fi
    grep -o '"skill":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/"skill":"//;s/"$//' || true
}

# =============================================================================
# Advanced Session Analysis
# =============================================================================

# Analyze workflow sequence in session
# Usage: analyze_workflow_sequence "$session_file"
# Returns: ordered list of tool invocations
analyze_workflow_sequence() {
    local session_file="$1"
    
    if [ ! -f "$session_file" ]; then
        echo "[]"
        return 1
    fi
    
    # Extract tool invocations in order
    local sequence=""
    if command -v jq &> /dev/null; then
        sequence=$(jq -s '[.[] | select(.message.content != null) | .message.content[]? | select(.type == "tool_use") | {tool: .name, input: .input}]' "$session_file" 2>/dev/null || echo "[]")
    else
        # Fallback: grep for tool names
        sequence=$(grep -o '"name":"[^"]*"' "$session_file" 2>/dev/null | \
            grep -v '"name":"user"' | \
            grep -v '"name":"assistant"' | \
            sed 's/"name":"//;s/"$//' | \
            awk '{tools = tools "\"" $0 "\","} END {print "[" substr(tools, 1, length(tools)-1) "]"}' || echo "[]")
    fi
    
    echo "$sequence"
}

# Analyze tool call chain in session
# Usage: analyze_tool_chain "$session_file"
# Returns: tool call sequence with timestamps
analyze_tool_chain() {
    local session_file="$1"
    
    if [ ! -f "$session_file" ]; then
        echo "No session file"
        return 1
    fi
    
    echo "=== Tool Call Chain ==="
    
    local line_num=0
    local tool_count=0
    local skill_invoked=false
    local skill_line=0
    
    while IFS= read -r line; do
        ((line_num++)) || true
        
        # Check for Skill invocation
        if echo "$line" | grep -q '"name":"Skill"'; then
            skill_invoked=true
            skill_line=$line_num
            local skill_name=$(echo "$line" | grep -o '"skill":"[^"]*"' | sed 's/"skill":"//;s/"$//' | head -1)
            echo "  [SKILL] Line $line_num: $skill_name"
            ((tool_count++)) || true
        fi
        
        # Check for other tool invocations
        if echo "$line" | grep -q '"type":"tool_use"'; then
            local tool_name=$(echo "$line" | grep -o '"name":"[^"]*"' | head -1 | sed 's/"name":"//;s/"$//')
            if [ -n "$tool_name" ] && [ "$tool_name" != "Skill" ]; then
                local marker="TOOL"
                if $skill_invoked && [ $line_num -lt $skill_line ]; then
                    marker="PREMATURE"
                fi
                echo "  [$marker] Line $line_num: $tool_name"
                ((tool_count++)) || true
            fi
        fi
        
    done < "$session_file"
    
    echo ""
    echo "Total tool calls: $tool_count"
}

# Analyze cost breakdown by subagent (requires jq)
# Usage: analyze_cost_breakdown "$session_file"
analyze_cost_breakdown() {
    local session_file="$1"
    
    if [ ! -f "$session_file" ]; then
        echo "[ERROR] Session file not found"
        return 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo "[ERROR] jq is required for cost analysis"
        return 1
    fi
    
    echo "=== Cost Breakdown Analysis ==="
    echo ""
    
    # Main session usage
    local main_usage=$(jq -s '[.[] | select(.type == "result")] | last | .usage // {}' "$session_file" 2>/dev/null)
    
    if [ -n "$main_usage" ] && [ "$main_usage" != "{}" ]; then
        echo "Main Session:"
        echo "$main_usage" | jq -r '"  Input: \(.input_tokens // 0), Output: \(.output_tokens // 0), Cache Read: \(.cache_read_input_tokens // 0)"' 2>/dev/null || echo "  (unable to parse)"
        echo ""
    fi
    
    # Subagent usage
    local subagent_count=$(jq -s '[.[] | select(.type == "user") | .toolUseResult? | select(. != null) | select(.agentId != null)] | length' "$session_file" 2>/dev/null || echo "0")
    
    if [ "$subagent_count" -gt 0 ]; then
        echo "Subagents ($subagent_count total):"
        jq -s '[.[] | select(.type == "user") | .toolUseResult? | select(. != null) | select(.agentId != null)] | 
            .[] | {agentId, input: .usage.input_tokens // 0, output: .usage.output_tokens // 0, prompt: .prompt[0:50]}' \
            "$session_file" 2>/dev/null | \
            jq -r '"  \(.agentId): Input=\(.input), Output=\(.output) - \(.prompt)..."'
        echo ""
    fi
    
    # Total estimation (approximate pricing)
    local total_input=$(jq -s '[.[] | .usage.input_tokens // 0, .toolUseResult?.usage?.input_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    local total_output=$(jq -s '[.[] | .usage.output_tokens // 0, .toolUseResult?.usage?.output_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    local total_cache=$(jq -s '[.[] | .usage.cache_read_input_tokens // 0] | add' "$session_file" 2>/dev/null || echo "0")
    
    echo "Totals:"
    echo "  Input tokens: $total_input"
    echo "  Output tokens: $total_output"
    echo "  Cache read tokens: $total_cache"
    echo ""
}

# =============================================================================
# Token Usage Analysis
# =============================================================================

# Extract token usage from session file (requires jq)
# Usage: extract_token_usage "$session_file"
extract_token_usage() {
    local session_file="$1"
    if [ ! -f "$session_file" ]; then
        echo "{}"
        return 1
    fi

    if command -v jq &> /dev/null; then
        # Extract usage from the last result message
        jq -s '[.[] | select(.type == "result")] | last | .usage // {}' "$session_file" 2>/dev/null || echo "{}"
    else
        echo "{}"
    fi
}

# =============================================================================
# Test Result Tracking
# =============================================================================
# These functions provide basic test tracking for individual test scripts.
# Note: run-tests.sh uses its own TEST_RESULTS array and print_summary/output_json
# functions for more detailed tracking. Individual test scripts should use these
# functions or implement their own tracking as needed.
# =============================================================================

# Initialize test tracking
# Usage: init_test_tracking
init_test_tracking() {
    TEST_PASSED=0
    TEST_FAILED=0
    TEST_SKIPPED=0
    TEST_START_TIME=$(date +%s)
}

# Record a test result
# Usage: record_test "pass" "test_name" ["duration"]
record_test() {
    local result="$1"
    local test_name="$2"
    local duration="${3:-0}"

    case "$result" in
        pass|PASS)  ((TEST_PASSED++)) ;;
        fail|FAIL)  ((TEST_FAILED++)) ;;
        skip|SKIP)  ((TEST_SKIPPED++)) ;;
    esac
}

# Print test summary
# Usage: print_test_summary
print_test_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - TEST_START_TIME))

    echo ""
    echo "========================================"
    echo -e " ${BOLD}Test Results Summary${NC}"
    echo "========================================"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $TEST_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TEST_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TEST_SKIPPED"
    echo "  Duration: ${duration}s"
    echo ""

    if [ "$TEST_FAILED" -gt 0 ]; then
        print_status_failed
        return 1
    else
        print_status_passed
        return 0
    fi
}

# Output test results as JSON (for CI)
# Usage: output_test_json > results.json
output_test_json() {
    local end_time=$(date +%s)
    local duration=$((end_time - TEST_START_TIME))

    cat <<EOF
{
  "status": "$([ "$TEST_FAILED" -gt 0 ] && echo "failed" || echo "passed")",
  "passed": $TEST_PASSED,
  "failed": $TEST_FAILED,
  "skipped": $TEST_SKIPPED,
  "duration": $duration,
  "timestamp": "$(date -Iseconds)"
}
EOF
}

# =============================================================================
# Utility Functions
# =============================================================================

# Print a test section header
# Usage: print_section "Section Name"
print_section() {
    local name="$1"
    echo ""
    echo "----------------------------------------"
    echo " $name"
    echo "----------------------------------------"
}

# Print test start banner
# Usage: print_test_banner "Test Name" ["description"]
print_test_banner() {
    local name="$1"
    local description="${2:-}"

    echo ""
    echo "========================================"
    echo " $name"
    echo "========================================"
    [ -n "$description" ] && echo "" && echo "$description"
    echo ""
}

# =============================================================================
# Export Functions
# =============================================================================

export -f detect_platforms
export -f is_platform_available
export -f get_platform_version
export -f create_test_project
export -f cleanup_test_project
export -f run_claude
export -f run_opencode
export -f run_ai
export -f assert_contains
export -f assert_not_contains
export -f assert_count
export -f assert_order
export -f assert_file_exists
export -f check_file_links
export -f extract_agent_skills
export -f run_behavior_test
export -f analyze_premature_actions
export -f get_all_skills
export -f get_all_agents
export -f get_all_teams
export -f validate_skill_structure
export -f validate_skill_content
export -f validate_agent_structure
export -f validate_agent_content
export -f extract_team_skills
export -f validate_team_structure
export -f validate_team_content
export -f find_recent_session
export -f verify_skill_invoked
export -f verify_agent_dispatched
export -f count_tool_invocations
export -f check_premature_action
export -f get_triggered_skills
export -f analyze_workflow_sequence
export -f analyze_tool_chain
export -f analyze_cost_breakdown
export -f extract_token_usage
export -f init_test_tracking
export -f record_test
export -f print_test_summary
export -f output_test_json
export -f print_section
export -f print_test_banner
export -f setup_colors
export -f enable_colors
export -f disable_colors
export -f print_pass
export -f print_fail
export -f print_skip
export -f print_info
export -f print_warn
export -f print_error
export -f print_section_header
export -f print_status_passed
export -f print_status_failed
export LIB_DIR TESTS_DIR SKILLS_DIR

# Force enable colors when sourced (unless NO_COLOR is set)
if [ -z "${NO_COLOR:-}" ]; then
    FORCE_COLOR=1
    setup_colors
fi