#!/bin/bash
# Exp 9 chain: sim7 attribution runs -> exact classification + true CF -> analysis -> finalize. Resumes from per-run checkpoints in runs9/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs9 out9
echo "chain start $(date)" >> runs9_chain.log
$PY exp9_attr.py --procs 4 >> runs9_attr.log 2>&1
N=$(ls runs9/attr_*.json 2>/dev/null | wc -l)
echo "attr runs: $N / 7 at $(date)" >> runs9_chain.log
[ "$N" -ge 7 ] || { echo "attr incomplete $(date)" >> runs9_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f exp9_cf.py ] && [ -f analyze9.py ] && [ -f finalize9.sh ] && break; sleep 30; done
$PY exp9_cf.py --procs 4 >> runs9_cf.log 2>&1
N=$(ls runs9/cf9_*.pkl 2>/dev/null | wc -l)
echo "cf pickles: $N / 21 at $(date)" >> runs9_chain.log
[ "$N" -ge 21 ] || { echo "cf incomplete $(date)" >> runs9_chain.log; exit 1; }
$PY analyze9.py > runs9_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs9_chain.log; exit 1; }
echo "chain done $(date)" >> runs9_chain.log
[ -x ./finalize9.sh ] && ./finalize9.sh >> runs9_chain.log 2>&1
