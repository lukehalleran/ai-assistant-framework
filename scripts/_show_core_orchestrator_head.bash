#!/usr/bin/env bash
set -euo pipefail
echo '== core/orchestrator.py (first 260 lines) =='
nl -ba core/orchestrator.py 2>/dev/null | sed -n '1,260p' || true
