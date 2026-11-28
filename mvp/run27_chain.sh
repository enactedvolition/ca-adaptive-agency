#!/bin/bash
# Exp 27: assay validation on known-feedback systems. Rerun to resume (finished pickles / ES checkpoints are skipped / resumed).
cd "$(dirname "$0")"; source ../.venv/bin/activate; mkdir -p runs27
HAND=P0.25,P0.5,P1,P2,P4,P8,P16,P0,CONST0.8,STEP,SHAM_STEP,SHAM_P4
( [ -f runs27/es_es.npz ] || python exp27_run.py es --name es >> runs27/es_es.log 2>&1 ) &
( [ -f runs27/es_esblind.npz ] || python exp27_run.py es --name esblind >> runs27/es_esblind.log 2>&1 ) &
python exp27_run.py assay --procs 3 --systems $HAND >> runs27/assay.log 2>&1
python exp27_run.py assay --procs 3 --systems P0,P1,P4,P16,STEP --reps 0 --haz 0 >> runs27/assay.log 2>&1
wait
python exp27_run.py assay --procs 5 --systems ES,ES_ABL,ES_BLIND >> runs27/assay.log 2>&1
python exp27_run.py pay --procs 5 >> runs27/pay.log 2>&1
echo done > runs27/done.txt
