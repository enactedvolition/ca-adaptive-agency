#!/bin/bash
# Exp 11 chain: sim9 mass-claim dose-response runs (Price on build/claim/premium) -> exact classification + true CF -> analysis -> finalize. Resumes from checkpoints in runs11/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
P=${P:-3}
mkdir -p runs11 out11
echo "chain start $(date) procs=$P" >> runs11_chain.log
[ -f runs11/bitcheck.json ] && grep -q '"bit_for_bit": true' runs11/bitcheck.json || $PY exp11_run.py --bitcheck --steps 300 >> runs11_run.log 2>&1
grep -q '"bit_for_bit": true' runs11/bitcheck.json || { echo "bitcheck FAILED $(date)" >> runs11_chain.log; exit 1; }
$PY exp11_run.py --procs $P >> runs11_run.log 2>&1
N=$(ls runs11/run11_*.pkl 2>/dev/null | wc -l)
echo "runs: $N / 6 at $(date)" >> runs11_chain.log
[ "$N" -ge 6 ] || { echo "runs incomplete $(date)" >> runs11_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f exp11_cf.py ] && [ -f analyze11.py ] && [ -f finalize11.sh ] && break; sleep 30; done
$PY exp11_cf.py --procs $P >> runs11_cf.log 2>&1
N=$(ls runs11/cf11_*.pkl 2>/dev/null | wc -l)
echo "cf pickles: $N / 18 at $(date)" >> runs11_chain.log
[ "$N" -ge 18 ] || { echo "cf incomplete $(date)" >> runs11_chain.log; exit 1; }
$PY analyze11.py > runs11_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs11_chain.log; exit 1; }
echo "chain done $(date)" >> runs11_chain.log
[ -x ./finalize11.sh ] && ./finalize11.sh >> runs11_chain.log 2>&1
