#!/bin/bash
# Exp 12 chain: seed extension (seeds 3-8) for sw_full, sw_mg2, sw_mg2_shuffled. Resumes from checkpoints in runs12/.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs12 out12
echo "chain start $(date)" >> runs12_chain.log
for B in 3,4,5 6,7,8; do EXP12_SEEDS=$B $PY exp12_run.py --procs 3 >> runs12_run.log 2>&1; done
N=$(ls runs12/run12_*.pkl 2>/dev/null | wc -l)
echo "runs: $N / 6 at $(date)" >> runs12_chain.log
[ "$N" -ge 6 ] || { echo "runs incomplete $(date)" >> runs12_chain.log; exit 1; }
echo "runs done $(date)" >> runs12_chain.log
