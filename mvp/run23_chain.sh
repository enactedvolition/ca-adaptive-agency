#!/bin/bash
# Exp 23: per-channel decomposition of A (needs runs20/, runs17/, runs22/ incl. sham_forks and Exp 22 pickles). Rerun to resume (finished pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs23
python exp23_run.py --procs 4 >> runs23/perturb.log 2>&1
echo done > runs23/done.txt
