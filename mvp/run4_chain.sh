#!/bin/bash
cd /workspace/ca-agency-viz/mvp; source ../.venv/bin/activate
while [ $(ls runs4/price_*.pkl 2>/dev/null | wc -l) -lt 7 ]; do sleep 15; done
python exp4_perturb.py --procs 8 > runs4_perturb.log 2>&1
