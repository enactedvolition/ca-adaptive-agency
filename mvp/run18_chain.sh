#!/bin/bash
# Exp 18 follow-ups: parity swap at p*, then post hoc gate vs every other grid p (no swap). 3 procs, seeds split. Resumable.
cd "$(dirname "$0")"; PY=../.venv/bin/python
for S in 0,1,2 3,4,5 6,7,8; do
 ( $PY exp18_run.py --gate --seeds $S --swap 1 ; for P in 0.0025 0.0075 0.01 0.015 0.02 0.03 0.05; do $PY exp18_run.py --gate --seeds $S --gatep $P; done ) > runs18/chain_$S.log 2>&1 &
done
wait; date > runs18/chain_done.txt
