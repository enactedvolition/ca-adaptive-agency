"""Experiment 17 evolution arms (run only because the pre-registered mixed-competition gate passed; NIGHTLY_LOG 2026-11-19).
Same driver as exp16_run.run (exact Price decomposition with residual, 500-step checkpoints that resume bit-for-bit, forks at
t=2500 and 3000) with the simulator swapped to sim13 and sw_pays / sw_pays_blind given rep_fixed = 1 (ungated fixed-cost reproduction).
Outputs go to runs17/ (file names keep exp16_run's "run16_/ck16_/fork16_" prefixes; the directory distinguishes them).
  EXP16_SEEDS=0,1,2 python exp17_evo.py sw_pays        (one variant per process; rerun the same command to resume from checkpoint)"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["EXP16_OUT"] = os.path.join(HERE, "runs17")
sys.path.insert(0, HERE)
import sim13
sys.modules["sim12"] = sim13          # exp16_run.run does `import sim12` inside; route it to sim13
import exp16_run as E16
E16.VARIANTS = dict(sw_full=dict(sw_period=E16.SW), sw_pays=dict(sw_period=E16.SW, rep_free=1.0, rep_fixed=1.0),
                    sw_pays_blind=dict(sw_period=E16.SW, rep_free=1.0, rep_fixed=1.0, blind=1.0))
if __name__ == "__main__":
    os.makedirs(E16.OUT, exist_ok=True)
    for v in sys.argv[1:]:
        print("DONE", E16.TAG, E16.run(v, prefix="run17"), time.strftime("%H:%M:%S"), flush=True)
