#!/bin/bash
# Exp 14 chain: Exp 5 perturbation protocol (phase A) then Exp 6 energy-drop test (phase B) on the Exp 13 forks,
# sw_full / sw_mg2, seeds 0-8, t=10000 then t=5000. Resumes: finished (arm, seed, horizon, phase) pickles in runs14/ are skipped.
cd /workspace/ca-agency-viz/mvp
PY=../.venv/bin/python
mkdir -p runs14 out14
echo "chain start $(date)" >> runs14_chain.log
$PY exp14_perturb.py --procs 3 --phases A,B >> runs14_run.log 2>&1
NA=$(ls runs14/p14A_*.pkl 2>/dev/null | wc -l); NB=$(ls runs14/p14B_*.pkl 2>/dev/null | wc -l)
echo "runs: A $NA / 36, B $NB / 36 at $(date)" >> runs14_chain.log
[ "$NA" -ge 36 ] && [ "$NB" -ge 36 ] || { echo "runs incomplete $(date)" >> runs14_chain.log; exit 1; }
echo "runs done $(date)" >> runs14_chain.log
