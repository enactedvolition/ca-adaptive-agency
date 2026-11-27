#!/bin/bash
# Exp 25: starvation-predation pay-check (needs runs20/ forks). Rerun to resume (finished pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs25
python exp25_run.py --procs 6 --haz 0.004,0.0 >> runs25/paycheck.log 2>&1
echo done > runs25/done.txt
