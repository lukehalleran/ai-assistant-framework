#!/usr/bin/env bash
set -euo pipefail
echo '== core/dependencies.py (first 220 lines) =='
nl -ba core/dependencies.py 2>/dev/null | sed -n '1,220p' || true
