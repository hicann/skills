#!/bin/bash
set -e

show_help() {
    cat << 'EOF'
Ascend C Operator Development Environment Installer

Usage: init.sh [level]

Arguments:
  level   - Installation level: "project" (default) or "global"

Options:
  --help  - Show this help message

Examples:
  init.sh              # Project-level installation
  init.sh project      # Project-level installation
  init.sh global       # Global-level installation

Installation paths:
  Project: .opencode/skills/
  Global:    ~/.config/opencode/skills/
EOF
}

LEVEL="${1:-project}"

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

if [ "$LEVEL" = "global" ]; then
    SKILLS_ROOT_DIR="$HOME/.config/opencode/skills"
    OPENCODE_CONFIG_DIR="$HOME/.config/opencode"
else
    SKILLS_ROOT_DIR="$PROJECT_ROOT/.opencode/skills"
    OPENCODE_CONFIG_DIR="$PROJECT_ROOT/.opencode"
fi

echo "Installing Ascend C Operator Development Environment..."
echo "  Level: $LEVEL"
echo "  Skills: $SKILLS_ROOT_DIR"
echo "  Config: $OPENCODE_CONFIG_DIR"
echo ""

SOURCE_SKILLS_DIR="$ASCEND_AGENT_ROOT/skills"
echo "Installing skills from $SOURCE_SKILLS_DIR..."

if [ -d "$SOURCE_SKILLS_DIR" ]; then
    for skill_dir in "$SOURCE_SKILLS_DIR"/*; do
        if [ -d "$skill_dir" ]; then
            skill_name=$(basename "$skill_dir")
            target_dir="$SKILLS_ROOT_DIR/$skill_name"
            
            if [ -e "$target_dir" ]; then
                rm -rf "$target_dir"
                echo "  Removed existing: $skill_name"
            fi
            mkdir -p "$SKILLS_ROOT_DIR"
            cp -r "$(realpath "$skill_dir")" "$target_dir"
            echo "  Copied: $skill_name"
        fi
    done
    echo ""
    echo "Skills installation complete!"
else
    echo "  Warning: $SOURCE_SKILLS_DIR not found, skipping skills installation"
fi

echo ""
echo "Installing AGENTS.md..."
mkdir -p "$OPENCODE_CONFIG_DIR"
cp "$PROJECT_ROOT/AGENTS.md" "$OPENCODE_CONFIG_DIR/AGENTS.md"
echo "  Copied: AGENTS.md -> $OPENCODE_CONFIG_DIR/AGENTS.md"

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