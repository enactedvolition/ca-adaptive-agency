#!/bin/bash
# Exp 5 chain: Price runs -> perturbation forks -> analysis -> finalize. Every stage resumes from per-file checkpoints.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
$PY exp5_price.py run --procs 5 >> runs5_price.log 2>&1
[ $(ls runs5/price_*.pkl 2>/dev/null | wc -l) -ge 7 ] || { echo "price incomplete $(date)" >> runs5_chain.log; exit 1; }
$PY exp5_perturb.py --procs 5 >> runs5_perturb.log 2>&1
[ $(ls runs5/perturb/pert_*.pkl 2>/dev/null | wc -l) -ge 21 ] || { echo "perturb incomplete $(date)" >> runs5_chain.log; exit 1; }
$PY analyze5.py > runs5_analyze.log 2>&1 || { echo "analyze failed $(date)" >> runs5_chain.log; exit 1; }
echo "chain done $(date)" >> runs5_chain.log
[ -x ./finalize5.sh ] && ./finalize5.sh >> runs5_chain.log 2>&1
