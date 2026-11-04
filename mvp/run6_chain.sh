#!/bin/bash
# Exp 6 chain: regulated-variable forks -> analysis -> finalize. Resumes from per-run checkpoints in runs6/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
echo "chain start $(date)" >> runs6_chain.log
$PY exp6_regulate.py --procs 6 >> runs6_regulate.log 2>&1
[ $(ls runs6/reg_*.pkl 2>/dev/null | wc -l) -ge 21 ] || { echo "regulate incomplete $(date)" >> runs6_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f analyze6.py ] && [ -f finalize6.sh ] && break; sleep 30; done
$PY analyze6.py > runs6_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs6_chain.log; exit 1; }
echo "chain done $(date)" >> runs6_chain.log
[ -x ./finalize6.sh ] && ./finalize6.sh >> runs6_chain.log 2>&1
