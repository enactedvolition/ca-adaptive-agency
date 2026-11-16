#!/bin/bash
# Exp 15b chain (see NIGHTLY_LOG 2026-11-16): sw_full and sw_full_shuffled, seeds 0-8 as three batches of 3, 1-2 procs (PROCS env; 1 after an OOM tonight).
# Resumes: finished runs15/run15_*.pkl are skipped, partial runs continue from runs15/ck15_*.pkl (500-step checkpoints).
cd "$(dirname "$0")"
PY=${PY:-../../ca-agency-viz/.venv/bin/python}
mkdir -p runs15
for T in 0,1,2 3,4,5 6,7,8; do
  EXP15_SEEDS=$T $PY exp15_run.py --procs ${PROCS:-1} --jobs sw_full,sw_full_shuffled >> runs15/evo.log 2>&1
done
echo "evolution done $(date)" >> runs15/evo.log
