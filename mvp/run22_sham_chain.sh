#!/bin/bash
# Exp 22 optional age-matched CHECK: rerun Exp 19 SHAM saving the t=1500 state (runs22/sham_forks), then the Exp 22 births-off assay on it.
# Rerun to resume (exp19 skips finished runs; finished pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs22/sham_evo runs22/sham_forks
for S in 0,1,2 3,4,5 6,7,8; do
  ( EXP19_OUT=$PWD/runs22/sham_evo EXP19_FORKDIR=$PWD/runs22/sham_forks EXP19_SEEDS=$S python exp19_run.py sham >> runs22/sham_evo_$S.log 2>&1 ) &
done
wait
python exp22_run.py --procs 4 --pops sham >> runs22/perturb_sham.log 2>&1
echo done > runs22/done_sham.txt
