#!/bin/bash
set -e

show_help() {
    cat << 'EOF'
Ascend C Operator Development Environment Installer

Usage: init.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default) or "claude"

Options:
  --help  - Show this help message

Examples:
  init.sh                      # Project-level, OpenCode
  init.sh project opencode     # Project-level, OpenCode
  init.sh global claude        # Global-level, Claude Code
  init.sh project claude       # Project-level, Claude Code

Installation paths:
  OpenCode project: .opencode/skills/
  OpenCode global:  ~/.config/opencode/skills/
  Claude project:   .claude/skills/
  Claude global:    ~/.claude/skills/
EOF
}

LEVEL="${1:-project}"
TOOL="${2:-opencode}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ASCEND_AGENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$LEVEL" = "--help" ]; then
    show_help
    exit 0
fi

if [ "$LEVEL" != "global" ] && [ "$LEVEL" != "project" ]; then
    echo "Error: Invalid level '$LEVEL'. Must be 'global' or 'project'."
    exit 1
fi

if [ "$TOOL" != "opencode" ] && [ "$TOOL" != "claude" ]; then
    echo "Error: Invalid tool '$TOOL'. Must be 'opencode' or 'claude'."
    exit 1
fi

if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        SKILLS_ROOT_DIR="$HOME/.config/opencode/skills"
        OPENCODE_CONFIG_DIR="$HOME/.config/opencode"
    else
        SKILLS_ROOT_DIR="$HOME/.claude/skills"
        OPENCODE_CONFIG_DIR="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        SKILLS_ROOT_DIR="$PROJECT_ROOT/.opencode/skills"
        OPENCODE_CONFIG_DIR="$PROJECT_ROOT/.opencode"
    else
        SKILLS_ROOT_DIR="$PROJECT_ROOT/.claude/skills"
        OPENCODE_CONFIG_DIR="$PROJECT_ROOT/.claude"
    fi
fi

echo "Installing Ascend C Operator Development Environment..."
echo "  Level: $LEVEL"
echo "  Tool: $TOOL"
echo "  Skills: $SKILLS_ROOT_DIR"
echo "  Config: $OPENCODE_CONFIG_DIR"
echo ""

echo "Removing conflicting skill: ascendc-custom-op-template..."
for skill_root in "$PROJECT_ROOT/.opencode/skills" "$PROJECT_ROOT/.claude/skills" "$HOME/.config/opencode/skills" "$HOME/.claude/skills"; do
    conflicting_skill="$skill_root/ascendc-custom-op-template"
    if [ -e "$conflicting_skill" ]; then
        rm -rf "$conflicting_skill"
        echo "  Removed: $conflicting_skill"
    fi
done
echo ""

SOURCE_SKILLS_DIR="$ASCEND_AGENT_ROOT/skills"
echo "Installing skills from $SOURCE_SKILLS_DIR..."

if [ -d "$SOURCE_SKILLS_DIR" ]; then
    for skill_dir in "$SOURCE_SKILLS_DIR"/*; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            if [[ "$skill_name" == "ascendc-custom-op-template" ]]; then
                echo "  Skipping $skill_name (excluded)"
                continue
            fi
            target_dir="$SKILLS_ROOT_DIR/$skill_name"
            
            if [ -e "$target_dir" ]; then
                rm -rf "$target_dir"
                echo "  Removed existing: $skill_name"
            fi
            mkdir -p "$SKILLS_ROOT_DIR"
            ln -s "$(realpath "$skill_dir")" "$target_dir"
            echo "  Linked: $skill_name"
        fi
    done
    echo ""
    echo "Skills installation complete!"
else
    echo "  Warning: $SOURCE_SKILLS_DIR not found, skipping skills installation"
fi

echo ""
mkdir -p "$OPENCODE_CONFIG_DIR"

if [ "$TOOL" = "opencode" ]; then
    echo "Installing AGENTS.md..."
    ln -sf "$PROJECT_ROOT/AGENTS.md" "$OPENCODE_CONFIG_DIR/AGENTS.md"
    echo "  Linked: AGENTS.md -> $OPENCODE_CONFIG_DIR/AGENTS.md"
else
    echo "Installing CLAUDE.md..."
    ln -sf "$PROJECT_ROOT/AGENTS.md" "$OPENCODE_CONFIG_DIR/CLAUDE.md"
    echo "  Linked: CLAUDE.md -> $OPENCODE_CONFIG_DIR/CLAUDE.md"
fi

echo ""
echo "Cloning asc-devkit..."
ASC_DEVKIT_DIR="$SCRIPT_DIR/asc-devkit"

if [ -d "$ASC_DEVKIT_DIR" ]; then
    echo "  asc-devkit already exists, pulling latest changes..."
    cd "$ASC_DEVKIT_DIR"
    git pull
    cd "$SCRIPT_DIR"
else
    git clone https://gitcode.com/cann/asc-devkit.git "$ASC_DEVKIT_DIR"
    echo "  Cloned asc-devkit to $ASC_DEVKIT_DIR"
fi

echo ""
echo "Installation complete!"