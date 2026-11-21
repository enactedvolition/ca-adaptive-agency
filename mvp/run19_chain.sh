#!/bin/bash
# Exp 19: one process per arm, seed batches sequential; rerun to resume (finished runs skipped, checkpoints resumed).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs19
for arm in true sham; do
  ( for S in 0,1,2 3,4,5 6,7,8; do EXP19_SEEDS=$S python exp19_run.py $arm >> runs19/evo_${arm}.log 2>&1; done; echo done >> runs19/done_${arm}.txt ) &
done
wait
