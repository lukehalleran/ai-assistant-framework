#!/usr/bin/env bash
# Pre-agent session snapshot: captures git state, filesystem manifest, and
# selected untracked files so that accidental damage can be recovered.
#
# Usage:
#   bash scripts/agent_session_start.sh
#
# Output: .agent_snapshots/YYYYMMDD_HHMMSS/  with recovery artifacts.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(git rev-parse --show-toplevel)"
SNAP_BASE="${REPO_ROOT}/.agent_snapshots"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAP_DIR="${SNAP_BASE}/${TIMESTAMP}"
MAX_SNAPSHOTS=10
MAX_FILE_SIZE_BYTES=$((25 * 1024 * 1024))  # 25 MB

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Create snapshot directory
# ============================================================================

mkdir -p "$SNAP_DIR"

echo -e "${GREEN}[agent-safety]${NC} Creating snapshot: ${SNAP_DIR}"

# ============================================================================
# Git state (all read-only)
# ============================================================================

cd "$REPO_ROOT"

git branch --show-current > "${SNAP_DIR}/branch.txt" 2>/dev/null || echo "detached HEAD" > "${SNAP_DIR}/branch.txt"
git rev-parse HEAD > "${SNAP_DIR}/head.txt"
git status --short --untracked-files=normal > "${SNAP_DIR}/status.txt"
git diff > "${SNAP_DIR}/diff.patch" || true
git diff --cached > "${SNAP_DIR}/cached.diff" || true
git log --oneline -20 > "${SNAP_DIR}/reflog.txt" || true

# List untracked files (respecting .gitignore)
git ls-files --others --exclude-standard > "${SNAP_DIR}/untracked.txt" || true

echo -e "${GREEN}[agent-safety]${NC} Git state saved."

# ============================================================================
# Filesystem manifest via Python utility
# ============================================================================

python -m utils.fs_snapshot create "$REPO_ROOT" "${SNAP_DIR}/manifest.json"

echo -e "${GREEN}[agent-safety]${NC} Filesystem manifest saved."

# ============================================================================
# Untracked files tarball (filtered, size-capped)
# ============================================================================

UNTRACKED_LIST="${SNAP_DIR}/untracked.txt"
TAR_INPUT=$(mktemp)
SKIPPED_FILE="${SNAP_DIR}/skipped_untracked.txt"
: > "$TAR_INPUT"
: > "$SKIPPED_FILE"

while IFS= read -r file; do
    # Skip excluded patterns
    case "$file" in
        .agent_snapshots/*|RECOVERY_*|docs/.Rhistory) continue ;;
        *.rpm|*.zip|*.tar|*.tar.gz|*.7z) continue ;;
    esac

    # Skip files larger than threshold
    if [ -f "$file" ]; then
        file_size=$(stat --printf='%s' "$file" 2>/dev/null || stat -f '%z' "$file" 2>/dev/null || echo 0)
        if [ "$file_size" -gt "$MAX_FILE_SIZE_BYTES" ]; then
            echo "$file ($(( file_size / 1024 / 1024 ))MB)" >> "$SKIPPED_FILE"
            continue
        fi
        echo "$file" >> "$TAR_INPUT"
    fi
done < "$UNTRACKED_LIST"

if [ -s "$TAR_INPUT" ]; then
    tar czf "${SNAP_DIR}/untracked_files.tar.gz" -T "$TAR_INPUT" 2>/dev/null || true
    file_count=$(wc -l < "$TAR_INPUT")
    echo -e "${GREEN}[agent-safety]${NC} Archived ${file_count} untracked files."
else
    echo -e "${GREEN}[agent-safety]${NC} No untracked files to archive."
fi

if [ -s "$SKIPPED_FILE" ]; then
    skipped_count=$(wc -l < "$SKIPPED_FILE")
    echo -e "${YELLOW}[agent-safety]${NC} Skipped ${skipped_count} large files (see skipped_untracked.txt)."
fi

rm -f "$TAR_INPUT"

# ============================================================================
# Rotation: keep only latest MAX_SNAPSHOTS
# ============================================================================

# List snapshot dirs sorted oldest first, delete extras
snap_count=$(find "$SNAP_BASE" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$snap_count" -gt "$MAX_SNAPSHOTS" ]; then
    excess=$(( snap_count - MAX_SNAPSHOTS ))
    find "$SNAP_BASE" -mindepth 1 -maxdepth 1 -type d | sort | head -n "$excess" | while read -r old_dir; do
        echo -e "${YELLOW}[agent-safety]${NC} Removing old snapshot: $(basename "$old_dir")"
        rm -rf "$old_dir"
    done
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}[agent-safety]${NC} Snapshot complete: ${SNAP_DIR}"
echo -e "${GREEN}[agent-safety]${NC} Branch: $(cat "${SNAP_DIR}/branch.txt")"
echo -e "${GREEN}[agent-safety]${NC} HEAD:   $(cat "${SNAP_DIR}/head.txt" | head -c 12)"
echo -e "${GREEN}[agent-safety]${NC} To audit later: bash scripts/agent_session_audit.sh"
