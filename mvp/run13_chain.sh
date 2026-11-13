#!/bin/bash
# Exp 13 chain: 10000-step horizon, seeds 0-8 (3 batches of 3), sw_full / sw_mg2 / sw_mg2_shuffled. Resumes from runs13/ck13_* checkpoints.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs13 out13
echo "chain start $(date)" >> runs13_chain.log
for B in 0,1,2 3,4,5 6,7,8; do EXP13_SEEDS=$B $PY exp13_run.py --procs 3 >> runs13_run.log 2>&1; done
N=$(ls runs13/run13_*.pkl 2>/dev/null | wc -l)
echo "runs: $N / 9 at $(date)" >> runs13_chain.log
[ "$N" -ge 9 ] || { echo "runs incomplete $(date)" >> runs13_chain.log; exit 1; }
echo "runs done $(date)" >> runs13_chain.log
