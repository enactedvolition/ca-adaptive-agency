#!/bin/bash
# Exp 8 chain: threshold sweep + CF replay -> analysis -> finalize. Resumes from per-run checkpoints in runs8/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs8 out8
echo "chain start $(date)" >> runs8_chain.log
# Prefer finishing tonight; 4 procs to avoid CPU thrash (exp6 taught us 6 procs + jax oversubscribes).
$PY exp8_counterfactual.py --procs 4 >> runs8_cf.log 2>&1
N=$(ls runs8/cf_*.pkl 2>/dev/null | wc -l)
echo "cf pickles: $N / 21 at $(date)" >> runs8_chain.log
[ "$N" -ge 21 ] || { echo "cf incomplete $(date)" >> runs8_chain.log; exit 1; }
for i in $(seq 1 60); do [ -f analyze8.py ] && [ -f finalize8.sh ] && break; sleep 30; done
$PY analyze8.py > runs8_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs8_chain.log; exit 1; }
echo "chain done $(date)" >> runs8_chain.log
[ -x ./finalize8.sh ] && ./finalize8.sh >> runs8_chain.log 2>&1
