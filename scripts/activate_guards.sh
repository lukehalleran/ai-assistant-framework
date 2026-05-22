#!/usr/bin/env bash
# Activate shell command guards for the current session.
#
# Usage:
#   source scripts/activate_guards.sh
#
# After sourcing, rm/mv/rmdir/chmod/truncate/find commands are intercepted
# and checked against protected paths before executing. Safe commands pass
# through transparently. Destructive commands on protected paths are blocked.
#
# To make permanent, add to ~/.bashrc or ~/.zshrc:
#   source /home/lukeh/Daemon_v1/scripts/activate_guards.sh

GUARD_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/bin" 2>/dev/null && pwd)"

if [ -z "$GUARD_BIN" ] || [ ! -d "$GUARD_BIN" ]; then
    echo "[guard] ERROR: Could not find scripts/bin/ directory."
    return 1 2>/dev/null || exit 1
fi

if [[ ":$PATH:" != *":$GUARD_BIN:"* ]]; then
    export PATH="$GUARD_BIN:$PATH"
    echo "[guard] Shell command guards activated. Wrapped: rm, mv, rmdir, chmod, truncate, find"
    echo "[guard] Protected paths: data/, config/, core/, memory/, scripts/, .git/, etc."
    echo "[guard] To deactivate: export PATH=\"\${PATH#$GUARD_BIN:}\""
else
    echo "[guard] Shell command guards already active."
fi

# Activate Python filesystem guard in subprocesses via usercustomize.py.
# scripts/bin/ contains usercustomize.py which auto-activates the guard
# in child Python interpreters.
if [[ ":$PYTHONPATH:" != *":$GUARD_BIN:"* ]]; then
    export PYTHONPATH="${GUARD_BIN}${PYTHONPATH:+:$PYTHONPATH}"
fi
