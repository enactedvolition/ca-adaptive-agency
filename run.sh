#!/usr/bin/env bash
# Mac/Linux: run a short demo sim (~30 s CPU) and open the result.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
.venv/bin/python -m mvp.smoke --demo --out out_demo "$@"
F=out_demo/demo.gif
if command -v open >/dev/null 2>&1; then open "$F"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$F" >/dev/null 2>&1 || true
else echo "Open $F to view."; fi
