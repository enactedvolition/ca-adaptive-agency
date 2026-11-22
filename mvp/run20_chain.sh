#!/bin/bash
# Exp 20: (1) rerun Exp 19 TRUE with the t=1500 state saved (3 seed batches in parallel), (2) perturbation recovery per seed. Rerun to resume.
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs20/evo
for S in 0,1,2 3,4,5 6,7,8; do
  ( EXP19_OUT=$PWD/runs20/evo EXP19_FORKDIR=$PWD/runs20 EXP19_SEEDS=$S python exp19_run.py true >> runs20/evo_$S.log 2>&1 ) &
done
wait
python exp20_run.py --procs 4 >> runs20/perturb.log 2>&1
echo done > runs20/done.txt
