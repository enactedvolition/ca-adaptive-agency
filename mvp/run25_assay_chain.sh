#!/bin/bash
# Exp 25 pre-planned branch: assay positive control on the hand-coded COND population (needs runs20/ forks). Rerun to resume.
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs25
python exp25_assay.py --procs 5 >> runs25/assay.log 2>&1
echo done > runs25/assay_done.txt
