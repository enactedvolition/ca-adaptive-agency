#!/bin/bash
# Exp 24: gradient-conditioned open-loop movement control (needs runs20/, runs17/, runs22/ incl. sham_forks, Exp 22 and Exp 23 pickles). Rerun to resume (finished pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs24
python exp24_run.py --procs 4 >> runs24/perturb.log 2>&1
echo done > runs24/done.txt
