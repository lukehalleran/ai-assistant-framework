#!/usr/bin/env bash
# 2026-09-03 owner cleanup, in order: (1) purge 2 junk profile facts, (2) purge 67 error/box-test
# memory docs, (3) start the daemon. Each purge writes a pre-image backup first and refuses to
# run while a daemon is live. Stops at the first failure; the daemon only starts if both purges
# succeeded. Run from the repo root:  bash scripts/apply_20260903_cleanup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== 1/3 profile facts"
python scripts/purge_profile_facts.py --from-file data/profile_junk_candidates_20260903b.txt --apply
echo
echo "== 2/3 error / box-test memory docs"
python scripts/purge_error_memories.py --apply
echo
echo "== 3/3 starting daemon (Ctrl+C stops it, as usual)"
exec python main.py
