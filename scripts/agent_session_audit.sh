#!/usr/bin/env bash
# Post-agent session audit: compares current repo state to the latest
# (or specified) pre-agent snapshot. Read-only — safe to run repeatedly.
#
# Usage:
#   bash scripts/agent_session_audit.sh                # latest snapshot
#   bash scripts/agent_session_audit.sh /path/to/snap  # specific snapshot

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
SNAP_BASE="${REPO_ROOT}/.agent_snapshots"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Locate snapshot
# ============================================================================

if [ -n "${1:-}" ] && [ -d "$1" ]; then
    SNAP_DIR="$1"
else
    SNAP_DIR="$(find "$SNAP_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
    if [ -z "$SNAP_DIR" ]; then
        echo -e "${RED}[audit]${NC} No snapshots found in ${SNAP_BASE}"
        echo "       Run: bash scripts/agent_session_start.sh"
        exit 1
    fi
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} Agent Session Audit${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${GREEN}Snapshot:${NC} ${SNAP_DIR}"
echo ""

# ============================================================================
# Saved state
# ============================================================================

echo -e "${CYAN}--- Saved State ---${NC}"
if [ -f "${SNAP_DIR}/branch.txt" ]; then
    echo -e "  Branch: $(cat "${SNAP_DIR}/branch.txt")"
fi
if [ -f "${SNAP_DIR}/head.txt" ]; then
    echo -e "  HEAD:   $(head -c 12 "${SNAP_DIR}/head.txt")"
fi
echo ""

# ============================================================================
# Current state
# ============================================================================

cd "$REPO_ROOT"

echo -e "${CYAN}--- Current State ---${NC}"
echo -e "  Branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
echo -e "  HEAD:   $(git rev-parse HEAD | head -c 12)"
echo ""

# Branch change?
saved_branch=""
if [ -f "${SNAP_DIR}/branch.txt" ]; then
    saved_branch="$(cat "${SNAP_DIR}/branch.txt")"
fi
current_branch="$(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
if [ "$saved_branch" != "$current_branch" ]; then
    echo -e "${RED}[WARNING]${NC} Branch changed: ${saved_branch} -> ${current_branch}"
    echo ""
fi

# HEAD change?
saved_head=""
if [ -f "${SNAP_DIR}/head.txt" ]; then
    saved_head="$(cat "${SNAP_DIR}/head.txt")"
fi
current_head="$(git rev-parse HEAD)"
if [ "$saved_head" != "$current_head" ]; then
    echo -e "${YELLOW}[INFO]${NC} HEAD moved: $(echo "$saved_head" | head -c 12) -> $(echo "$current_head" | head -c 12)"
    echo "  New commits:"
    git log --oneline "${saved_head}..${current_head}" 2>/dev/null | sed 's/^/    /' || true
    echo ""
fi

# ============================================================================
# Git status
# ============================================================================

echo -e "${CYAN}--- Git Status ---${NC}"

modified=$(git diff --name-only 2>/dev/null)
staged=$(git diff --cached --name-only 2>/dev/null)
deleted=$(git diff --diff-filter=D --name-only 2>/dev/null)
untracked=$(git ls-files --others --exclude-standard 2>/dev/null)

if [ -n "$staged" ]; then
    echo -e "${YELLOW}  Staged files:${NC}"
    echo "$staged" | sed 's/^/    /'
    echo ""
fi

if [ -n "$modified" ]; then
    echo -e "${YELLOW}  Modified files:${NC}"
    echo "$modified" | sed 's/^/    /'
    echo ""
fi

if [ -n "$deleted" ]; then
    echo -e "${RED}  Deleted files:${NC}"
    echo "$deleted" | sed 's/^/    /'
    echo ""
fi

if [ -n "$untracked" ]; then
    echo -e "${YELLOW}  Untracked files:${NC}"
    echo "$untracked" | sed 's/^/    /'
    echo ""
fi

if [ -z "$modified" ] && [ -z "$staged" ] && [ -z "$deleted" ] && [ -z "$untracked" ]; then
    echo -e "  ${GREEN}Clean working tree.${NC}"
    echo ""
fi

# ============================================================================
# Manifest diff
# ============================================================================

if [ -f "${SNAP_DIR}/manifest.json" ]; then
    echo -e "${CYAN}--- Manifest Diff ---${NC}"

    CURRENT_MANIFEST=$(mktemp --suffix=.json)
    trap "rm -f '$CURRENT_MANIFEST'" EXIT

    python -m utils.fs_snapshot create "$REPO_ROOT" "$CURRENT_MANIFEST" > /dev/null 2>&1

    # Run diff via Python
    python -m utils.fs_snapshot diff "${SNAP_DIR}/manifest.json" "$CURRENT_MANIFEST"
    echo ""
else
    echo -e "${YELLOW}[audit]${NC} No saved manifest found — skipping manifest diff."
    echo ""
fi

# ============================================================================
# Large file check
# ============================================================================

echo -e "${CYAN}--- Suspicious Large Files ---${NC}"
large_files_found=false
while IFS= read -r file; do
    if [ -f "$file" ]; then
        size=$(stat --printf='%s' "$file" 2>/dev/null || stat -f '%z' "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt $((10 * 1024 * 1024)) ]; then
            echo -e "  ${YELLOW}$(( size / 1024 / 1024 ))MB${NC}  $file"
            large_files_found=true
        fi
    fi
done <<< "$(git ls-files --others --exclude-standard 2>/dev/null; git diff --name-only 2>/dev/null)"

if [ "$large_files_found" = false ]; then
    echo -e "  ${GREEN}None found.${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} Audit complete.${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Recovery artifacts in: ${SNAP_DIR}"
echo -e "  diff.patch          — unstaged changes at snapshot time"
echo -e "  cached.diff         — staged changes at snapshot time"
echo -e "  untracked_files.tar.gz — archived untracked files"
echo -e "  manifest.json       — full filesystem manifest"
