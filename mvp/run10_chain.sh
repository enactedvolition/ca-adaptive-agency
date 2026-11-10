#!/bin/bash
# Exp 10 chain: sim8 guard runs (Price on build) -> exact classification + true CF -> analysis -> finalize. Resumes from per-run checkpoints in runs10/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs10 out10
echo "chain start $(date)" >> runs10_chain.log
[ -f runs10/bitcheck.json ] && grep -q '"steps": 300' runs10/bitcheck.json || $PY exp10_run.py --bitcheck --steps 300 >> runs10_run.log 2>&1
$PY exp10_run.py --procs 6 >> runs10_run.log 2>&1
N=$(ls runs10/run10_*.pkl 2>/dev/null | wc -l)
echo "runs: $N / 6 at $(date)" >> runs10_chain.log
[ "$N" -ge 6 ] || { echo "runs incomplete $(date)" >> runs10_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f exp10_cf.py ] && [ -f analyze10.py ] && [ -f finalize10.sh ] && break; sleep 30; done
$PY exp10_cf.py --procs 6 >> runs10_cf.log 2>&1
N=$(ls runs10/cf10_*.pkl 2>/dev/null | wc -l)
echo "cf pickles: $N / 18 at $(date)" >> runs10_chain.log
[ "$N" -ge 18 ] || { echo "cf incomplete $(date)" >> runs10_chain.log; exit 1; }
$PY analyze10.py > runs10_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs10_chain.log; exit 1; }
echo "chain done $(date)" >> runs10_chain.log
[ -x ./finalize10.sh ] && ./finalize10.sh >> runs10_chain.log 2>&1
