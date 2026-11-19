"""Experiment 17 perturbation protocols = exp16_perturb.py unchanged (Exp 6 energy-drop phase B with PRIMARY B = r_cl - r_blind in
sw_pays at t=3000, n=9; Exp 5 kick phase A closed vs open-loop vs frozen input), with the simulator routed to sim13 and the arms' params
from exp17_evo (sw_pays / sw_pays_blind have rep_fixed = 1). Forks: runs17/fork16_{v}_s{s}_t{T}.npz. Output: runs17p/ (one pickle per
(arm, seed, horizon, phase) = the checkpoint; rerunning skips finished jobs). Primary jobs are queued first.
  python exp17_perturb.py --procs 2 --phases B,A --horizons 3000,2500"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("EXP16P_OUT", os.path.join(HERE, "runs17p")); os.environ.setdefault("EXP16_FORKS", os.path.join(HERE, "runs17"))
sys.path.insert(0, HERE)
import exp17_evo          # noqa: F401  (routes sim12 -> sim13 and patches exp16_run.VARIANTS; runs in spawned workers too)
import exp16_perturb as P
if __name__ == "__main__":
    import argparse, time, json, multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--phases", default="B,A"); ap.add_argument("--horizons", default="3000,2500")
    ap.add_argument("--variants", default=",".join(P.VARIANTS)); ap.add_argument("--seeds", default=",".join(map(str, P.SEEDS))); a = ap.parse_args()
    os.makedirs(P.OUT, exist_ok=True)
    jobs = [(v, s, tf, ph) for ph in a.phases.split(",") for tf in map(int, a.horizons.split(",")) for v in a.variants.split(",") for s in map(int, a.seeds.split(","))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w, d in pool.imap_unordered(P.worker, jobs):
            print("DONE", os.path.basename(p), f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", time.strftime("%H:%M:%S"), flush=True)
