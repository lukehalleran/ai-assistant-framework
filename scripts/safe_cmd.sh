#!/usr/bin/env bash
# Safe command wrapper: blocks destructive shell commands unless explicitly unlocked.
#
# Usage (prefix any command):
#   bash scripts/safe_cmd.sh rm -rf data/
#   bash scripts/safe_cmd.sh mv config/ /tmp/
#   bash scripts/safe_cmd.sh chmod 000 main.py
#
# Blocked by default:
#   rm -rf on protected dirs, mv of protected paths, chmod 000 on protected files,
#   find -delete on protected dirs, truncate on protected files, rmdir on protected dirs
#
# Always blocked (even with unlock):
#   rm -rf /, rm -rf ., rm -rf ~, rm -rf *
#
# Unlock mechanisms (same as safe_git.sh):
#   1. Environment variable:  ALLOW_DESTRUCTIVE_OPS=1 scripts/safe_cmd.sh rm -rf data/
#   2. One-shot lockfile:     touch .agent_allow_destructive_once
#                             scripts/safe_cmd.sh rm -rf data/
#                             (lockfile consumed after use)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LOCKFILE="${REPO_ROOT}/.agent_allow_destructive_once"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD_BIN_DIR="${SCRIPT_DIR}/bin"

# Resolve the real binary for a command, skipping our wrapper directory.
# This prevents infinite recursion when scripts/bin/ is on PATH.
_find_real_binary() {
    local cmd="$1"
    # Search standard system paths directly (fast, portable)
    for dir in /usr/bin /bin /usr/local/bin /usr/sbin /sbin; do
        [ -x "$dir/$cmd" ] && echo "$dir/$cmd" && return 0
    done
    # Fallback: search PATH, skipping our wrapper dir
    local IFS=':'
    for dir in $PATH; do
        local resolved
        resolved="$(cd "$dir" 2>/dev/null && pwd)" || continue
        [ "$resolved" = "$GUARD_BIN_DIR" ] && continue
        [ -x "$dir/$cmd" ] && echo "$dir/$cmd" && return 0
    done
    echo "$cmd"  # last resort
}

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Classification via Python
# ============================================================================

classify_cmd() {
    # Call Python classifier, returns JSON-like output.
    # Exit codes: 0=safe, 1=protected (blockable), 2=always blocked
    local args_json
    args_json=$(python3 -c "
import json, sys
args = sys.argv[1:]
sys.stdout.write(json.dumps(args))
" "$@")

    local result
    result=$(python3 -c "
import json, sys
sys.path.insert(0, '${REPO_ROOT}')
from utils.shell_cmd_guard import classify_shell_cmd
args = json.loads(sys.argv[1])
r = classify_shell_cmd(args, '${REPO_ROOT}')
if not r['destructive']:
    sys.exit(0)
print(r.get('reason', 'destructive operation'))
if r.get('severity') == 'always':
    sys.exit(2)
sys.exit(1)
" "$args_json" 2>&1)

    CLASSIFY_EXIT=$?
    CLASSIFY_REASON="$result"
}

# ============================================================================
# Unlock check (mirrors safe_git.sh)
# ============================================================================

check_unlock() {
    if [ "${ALLOW_DESTRUCTIVE_OPS:-}" = "1" ]; then
        return 0
    fi
    if [ -f "$LOCKFILE" ]; then
        return 0
    fi
    return 1
}

consume_lockfile() {
    if [ -f "$LOCKFILE" ]; then
        $(_find_real_binary rm) -f "$LOCKFILE"
        echo -e "${YELLOW}[safe-cmd]${NC} One-shot lockfile consumed."
    fi
}

# ============================================================================
# Main
# ============================================================================

if [ $# -eq 0 ]; then
    echo -e "${RED}[safe-cmd]${NC} No command provided."
    echo "Usage: scripts/safe_cmd.sh <command> [args...]"
    exit 1
fi

# If the command is git, pass through (use safe_git.sh for git)
CMD_BASE="$(basename "$1")"
REAL_CMD="$(_find_real_binary "$CMD_BASE")"

if [ "$CMD_BASE" = "git" ]; then
    echo -e "${CYAN}[safe-cmd]${NC} git commands should use scripts/safe_git.sh — passing through."
    "$REAL_CMD" "${@:2}"
    exit $?
fi

classify_cmd "$@"

case $CLASSIFY_EXIT in
    0)
        # Safe command — pass through (use real binary to avoid wrapper recursion)
        "$REAL_CMD" "${@:2}"
        ;;
    2)
        # Always blocked — no override possible
        echo -e "${RED}[safe-cmd]${NC} PERMANENTLY BLOCKED command:"
        echo -e "${RED}[safe-cmd]${NC}   $*"
        echo -e "${RED}[safe-cmd]${NC}   Reason: ${CLASSIFY_REASON}"
        echo ""
        echo -e "This operation is ${RED}always blocked${NC} regardless of unlock."
        echo -e "It would cause catastrophic data loss."
        exit 1
        ;;
    1)
        # Protected — blockable with unlock
        if check_unlock; then
            echo -e "${YELLOW}[safe-cmd]${NC} WARNING: Running destructive command with explicit unlock:"
            echo -e "${YELLOW}[safe-cmd]${NC}   $*"
            echo -e "${YELLOW}[safe-cmd]${NC}   Reason: ${CLASSIFY_REASON}"
            consume_lockfile
            "$REAL_CMD" "${@:2}"
        else
            echo -e "${RED}[safe-cmd]${NC} BLOCKED destructive command:"
            echo -e "${RED}[safe-cmd]${NC}   $*"
            echo -e "${RED}[safe-cmd]${NC}   Reason: ${CLASSIFY_REASON}"
            echo ""
            echo -e "To proceed, use one of:"
            echo -e "  ${GREEN}ALLOW_DESTRUCTIVE_OPS=1${NC} scripts/safe_cmd.sh $*"
            echo -e "  ${GREEN}touch .agent_allow_destructive_once${NC} && scripts/safe_cmd.sh $*"
            exit 1
        fi
        ;;
    *)
        # Unexpected — fail open with warning
        echo -e "${YELLOW}[safe-cmd]${NC} WARNING: Classification failed (exit=$CLASSIFY_EXIT). Proceeding cautiously."
        "$REAL_CMD" "${@:2}"
        ;;
esac
