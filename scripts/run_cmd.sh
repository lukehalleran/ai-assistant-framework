#!/usr/bin/env bash
set -euo pipefail

# Executes the command provided in $RUN_CMD using bash -lc so that
# standard shell features (globbing, pipes) work as expected.
# Usage with wrapper:
#   RUN_CMD="rg -n system core" CMD="scripts/run_cmd.sh" \
#   bash --noprofile --norc -lc 'source ".venv/bin/activate" && "$CMD"'

if [[ -z "${RUN_CMD:-}" ]]; then
  echo "RUN_CMD is empty; nothing to run" >&2
  exit 2
fi

exec bash --noprofile --norc -lc "$RUN_CMD"

