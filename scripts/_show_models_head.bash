#!/usr/bin/env bash
set -euo pipefail
echo '== models/model_manager.py (first 260 lines) =='
nl -ba models/model_manager.py 2>/dev/null | sed -n '1,260p' || true
