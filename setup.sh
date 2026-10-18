#!/usr/bin/env bash
# Mac/Linux: create .venv, install deps, run the smoke test.  GPU (Linux/WSL2): ./setup.sh gpu
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then for c in python3.13 python3.12 python3.11 python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done; fi
[ -n "$PY" ] || { echo "Python 3.11+ not found. Install it (macOS: brew install python@3.12; Ubuntu: sudo apt install python3 python3-venv)"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || { echo "Need Python 3.11+, found $("$PY" -V)"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
REQ=requirements.txt; [ "${1:-}" = "gpu" ] && REQ=requirements-gpu.txt
.venv/bin/python -m pip install -q -r "$REQ"
.venv/bin/python -m mvp.smoke
echo "Setup done. Next: ./run.sh  (demo)  or  make test"
