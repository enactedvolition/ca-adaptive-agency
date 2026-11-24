#!/bin/bash
# Exp 22: births-off (targets' rep forced off in every arm) foraging-recovery assay on Exp 20 TRUE t=1500 forks (runs20/) and the no-implant
# Exp 17 sw_pays t=3000 forks (runs17/). Rerun to resume (finished (pop, seed) pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs22
python exp22_run.py --procs 4 >> runs22/perturb.log 2>&1
echo done > runs22/done.txt
