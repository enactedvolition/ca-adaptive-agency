#!/bin/bash
# Exp 26: (1) COND survival-assay replication (K=200, fresh rng), (2) survival re-score of TRUE/ctrl/SHAM populations. Needs runs20/, runs17/, runs22/sham_forks.
# Rerun to resume (finished pickles skipped).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs26
python exp26_cond.py --procs 4 >> runs26/cond.log 2>&1
python exp26_pop.py --procs 4 >> runs26/pop.log 2>&1
echo done > runs26/done.txt
