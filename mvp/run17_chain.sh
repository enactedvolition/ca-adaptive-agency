#!/bin/bash
# Exp 17 evolution: 3 arms x seeds 0-8 (vmapped batches of 3), one process per arm. Resumable: rerun to continue from checkpoints.
cd "$(dirname "$0")"; PY=../.venv/bin/python
for V in sw_pays sw_pays_blind sw_full; do
  ( for S in 0,1,2 3,4,5 6,7,8; do EXP16_SEEDS=$S $PY exp17_evo.py $V; done ) > runs17/evo_$V.log 2>&1 &
done
wait; echo "evolution done $(date)" > runs17/evo_done.txt
