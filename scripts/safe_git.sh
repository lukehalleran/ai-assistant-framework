#!/usr/bin/env bash
# Safe git wrapper: blocks destructive git commands unless explicitly unlocked.
#
# Usage (drop-in replacement for git):
#   bash scripts/safe_git.sh status
#   bash scripts/safe_git.sh diff
#   bash scripts/safe_git.sh log --oneline -5
#
# Blocked by default:
#   git restore, git reset --hard, git clean, git checkout -- <path>,
#   git switch -C, git branch -D, git push (all forms)
#
# Unlock mechanisms:
#   1. Environment variable:  ALLOW_DESTRUCTIVE_OPS=1 scripts/safe_git.sh restore .
#   2. One-shot lockfile:     touch .agent_allow_destructive_once
#                             scripts/safe_git.sh restore .
#                             (lockfile consumed after use)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
LOCKFILE="${REPO_ROOT}/.agent_allow_destructive_once"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# ============================================================================
# Classification
# ============================================================================

is_destructive() {
    # Returns 0 (true) if the git command is destructive.
    local args=("$@")
    local subcmd="${args[0]:-}"

    case "$subcmd" in
        restore)
            return 0
            ;;
        clean)
            return 0
            ;;
        push)
            return 0
            ;;
        reset)
            # Only block reset --hard / --merge / --keep
            for arg in "${args[@]}"; do
                case "$arg" in
                    --hard|--merge|--keep) return 0 ;;
                esac
            done
            return 1
            ;;
        checkout)
            # Block: checkout -- <path>, checkout <file-that-exists>
            # Allow: checkout <branch>, checkout -b <branch>
            for arg in "${args[@]:1}"; do
                case "$arg" in
                    -b|-B) return 1 ;;  # branch creation — safe
                    --)    return 0 ;;  # explicit path separator — destructive
                esac
            done
            # If there's exactly one arg after 'checkout' and it's a file, block it
            if [ "${#args[@]}" -eq 2 ]; then
                local target="${args[1]}"
                if [ -f "${REPO_ROOT}/${target}" ] && ! git rev-parse --verify "$target" &>/dev/null; then
                    return 0  # It's a file, not a branch
                fi
            fi
            return 1
            ;;
        switch)
            for arg in "${args[@]}"; do
                case "$arg" in
                    -C) return 0 ;;  # force-create (overwrites branch)
                esac
            done
            return 1
            ;;
        branch)
            for arg in "${args[@]}"; do
                case "$arg" in
                    -D) return 0 ;;  # force-delete
                esac
            done
            return 1
            ;;
    esac

    return 1  # Not destructive
}

# ============================================================================
# Unlock check
# ============================================================================

check_unlock() {
    # Returns 0 (true) if destructive ops are unlocked.

    # Env var unlock
    if [ "${ALLOW_DESTRUCTIVE_OPS:-}" = "1" ]; then
        return 0
    fi

    # One-shot lockfile unlock
    if [ -f "$LOCKFILE" ]; then
        return 0
    fi

    return 1
}

consume_lockfile() {
    if [ -f "$LOCKFILE" ]; then
        rm -f "$LOCKFILE"
        echo -e "${YELLOW}[safe-git]${NC} One-shot lockfile consumed."
    fi
}

# ============================================================================
# Main
# ============================================================================

if [ $# -eq 0 ]; then
    echo -e "${RED}[safe-git]${NC} No git arguments provided."
    echo "Usage: scripts/safe_git.sh <git-args>"
    exit 1
fi

if is_destructive "$@"; then
    if check_unlock; then
        echo -e "${YELLOW}[safe-git]${NC} WARNING: Running destructive command with explicit unlock:"
        echo -e "${YELLOW}[safe-git]${NC}   git $*"
        consume_lockfile
        git "$@"
    else
        echo -e "${RED}[safe-git]${NC} BLOCKED destructive command:"
        echo -e "${RED}[safe-git]${NC}   git $*"
        echo ""
        echo -e "To proceed, use one of:"
        echo -e "  ${GREEN}ALLOW_DESTRUCTIVE_OPS=1${NC} scripts/safe_git.sh $*"
        echo -e "  ${GREEN}touch .agent_allow_destructive_once${NC} && scripts/safe_git.sh $*"
        exit 1
    fi
else
    # Safe command — pass through
    git "$@"
fi
