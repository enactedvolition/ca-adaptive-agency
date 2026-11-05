#!/bin/bash
# Exp 7 chain: dark-room classification on Exp 5 forks -> analysis -> finalize. Resumes from per-run checkpoints in runs7/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
echo "chain start $(date)" >> runs7_chain.log
$PY exp7_darkroom.py --procs 6 >> runs7_darkroom.log 2>&1
[ $(ls runs7/dark_*.pkl 2>/dev/null | wc -l) -ge 21 ] || { echo "darkroom incomplete $(date)" >> runs7_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f analyze7.py ] && [ -f finalize7.sh ] && break; sleep 30; done
$PY analyze7.py > runs7_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs7_chain.log; exit 1; }
echo "chain done $(date)" >> runs7_chain.log
[ -x ./finalize7.sh ] && ./finalize7.sh >> runs7_chain.log 2>&1
