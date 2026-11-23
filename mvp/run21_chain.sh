#!/bin/bash
# Exp 21: Exp 20 perturbation-recovery assay on the no-implant Exp 17 sw_pays t=3000 forks (needs runs17/). Rerun to resume (finished seeds skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs21
python exp21_run.py --procs 4 >> runs21/perturb.log 2>&1
echo done > runs21/done.txt
