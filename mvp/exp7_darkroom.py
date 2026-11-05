"""Experiment 7: dark-room / niche-construction classification on Exp 5 forks (step 2500).

ARCHITECTURE.md §15 after "make PE matter": rerun RQ3 under pe_cost while depletion is still on
(the Friston question). Exp 5 built pe_cost but only ran Price + hidden-state recovery; Exp 6
changed the recovery yardstick and still got nulls. Tonight classifies every alive agent on the
existing Exp 5 forks with the same regime logic as metrics2.regimes (thresholds identical to v2).

  python exp7_darkroom.py --procs 6      # checkpoint = one file per run in runs7/
"""
from __future__ import annotations
import argparse, os, sys, time, pickle
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP7_OUT", os.path.join(HERE, "runs7"))
FORKS = os.path.join(HERE, "runs5")
VARIANTS = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
SEEDS = [0, 1, 2]
ENV_VAR = 0.35          # BASE in exp5_price.py
N_LAYOUTS = 4

def layout_index(variant, t, params):
    per = float(params.get("sw_period", 0.0) or 0.0)
    if per <= 0:
        return 0
    return int(np.floor(float(t) / per)) % N_LAYOUTS

def amaps_from_kmaps(Kmaps):
    Kmaps = np.asarray(Kmaps, float)
    return 0.3 + Kmaps / Kmaps.max(axis=(1, 2), keepdims=True)

def worker(job):
    variant, seed = job
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    sys.path.insert(0, HERE)
    path = os.path.join(OUT, f"dark_{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0, "cached"
    import sim5, metrics2 as M2, exp5_price as E5
    t0 = time.time()
    cfg = sim5.Cfg(N=2048, L=1.0, G=64)
    _, _, _, consts = sim5.make(cfg)
    Amaps = amaps_from_kmaps(consts["Kmaps"])
    prm = dict(E5.BASE, **E5.VARIANTS[variant])
    z = np.load(os.path.join(FORKS, f"fork_{variant}_s{seed}.npz"))
    t = int(np.asarray(z["t"]))
    li = layout_index(variant, t, prm)
    Amap = np.asarray(Amaps[li])
    alive = np.asarray(z["alive"]).astype(bool)
    pos = np.asarray(z["pos"], float)
    cs = cfg.L / cfg.G
    cell = np.clip((pos / cs).astype(int), 0, cfg.G - 1)
    cx, cy = cell[:, 0], cell[:, 1]
    Wl = np.asarray(z["Wl"], float)
    W_here = Wl[cx, cy]
    be = float(prm.get("build_eff", 1.0))
    a = alive
    snap = dict(
        uid=np.asarray(z["uid"])[a],
        res=np.asarray(z["res"], float)[a],
        spd=np.asarray(z["spd"], float)[a],
        own=np.asarray(z["own"], float)[a],
        cx=cx[a], cy=cy[a],
        W_here=W_here[a],
        build_eff=np.full(int(a.sum()), be, float),
    )
    v_term = cfg.a_max / (1.0 - cfg.damp)
    reg = M2.regimes(snap, cfg, ENV_VAR, Amap, v_term)
    n = len(reg)
    frac = {name: float(np.mean(reg == i)) if n else float("nan") for i, name in enumerate(M2.REGIMES)}
    dark_any = float(sum(frac[k] for k in M2.REGIMES if k.startswith("dark"))) if n else float("nan")
    resident = (snap["res"] >= 100) & (snap["spd"] < 0.2 * v_term)
    surp = np.asarray(z["surp"], float)[a]
    counts = {name: int((reg == i).sum()) for i, name in enumerate(M2.REGIMES)}
    out = dict(
        variant=variant, seed=seed, t=t, layout_index=li, env_var=ENV_VAR, build_eff=be,
        params=prm, n_alive=int(n), v_term=float(v_term),
        regime_frac=frac, regime_counts=counts, dark_any=dark_any,
        summaries=dict(
            mean_res=float(np.mean(snap["res"])) if n else float("nan"),
            median_res=float(np.median(snap["res"])) if n else float("nan"),
            mean_spd=float(np.mean(snap["spd"])) if n else float("nan"),
            median_spd=float(np.median(snap["spd"])) if n else float("nan"),
            mean_own=float(np.mean(snap["own"])) if n else float("nan"),
            median_own=float(np.median(snap["own"])) if n else float("nan"),
            mean_surp=float(np.mean(surp)) if n else float("nan"),
            median_surp=float(np.median(surp)) if n else float("nan"),
            wall_mass=float(Wl.sum()),
            frac_resident=float(np.mean(resident)) if n else float("nan"),
            mean_W_here=float(np.mean(snap["W_here"])) if n else float("nan"),
        ),
        wall_s=time.time() - t0,
    )
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, time.time() - t0, "ok"

def smoke():
    """Classify one fork and print regime counts (no checkpoint write unless --write)."""
    sys.path.insert(0, HERE)
    import sim5, metrics2 as M2, exp5_price as E5
    variant, seed = "sw_pecost_pe", 0
    cfg = sim5.Cfg(N=2048, L=1.0, G=64)
    _, _, _, consts = sim5.make(cfg)
    Amaps = amaps_from_kmaps(consts["Kmaps"])
    prm = dict(E5.BASE, **E5.VARIANTS[variant])
    z = np.load(os.path.join(FORKS, f"fork_{variant}_s{seed}.npz"))
    t = int(np.asarray(z["t"])); li = layout_index(variant, t, prm); Amap = np.asarray(Amaps[li])
    alive = np.asarray(z["alive"]).astype(bool)
    pos = np.asarray(z["pos"], float); cs = cfg.L / cfg.G
    cell = np.clip((pos / cs).astype(int), 0, cfg.G - 1)
    cx, cy = cell[:, 0], cell[:, 1]
    Wl = np.asarray(z["Wl"], float); a = alive
    snap = dict(uid=np.asarray(z["uid"])[a], res=np.asarray(z["res"], float)[a], spd=np.asarray(z["spd"], float)[a],
                own=np.asarray(z["own"], float)[a], cx=cx[a], cy=cy[a], W_here=Wl[cx, cy][a],
                build_eff=np.full(int(a.sum()), 1.0, float))
    v_term = cfg.a_max / (1.0 - cfg.damp)
    reg = M2.regimes(snap, cfg, ENV_VAR, Amap, v_term)
    print(f"smoke {variant} s{seed}: t={t} li={li} n_alive={len(reg)} v_term={v_term:.6f}")
    for i, name in enumerate(M2.REGIMES):
        print(f"  {name}: {(reg == i).sum()} ({(reg == i).mean():.4f})")
    print(f"  dark_any: {np.mean(np.isin(reg, [2, 3, 4])):.4f}")
    print(f"  wall_mass: {Wl.sum():.3f} mean_surp: {np.asarray(z['surp'], float)[a].mean():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        smoke(); sys.exit(0)
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    if a.variant is not None:
        jobs = [(a.variant, a.seed if a.seed is not None else 0)]
    t0 = time.time()
    if a.procs <= 1 or len(jobs) == 1:
        for j in jobs:
            p, w, d = worker(j); print("DONE", os.path.basename(p), f"{w:.2f}s", d, flush=True)
    else:
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for p, w, d in pool.imap_unordered(worker, jobs):
                print("DONE", os.path.basename(p), f"{w:.2f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"all done in {time.time()-t0:.1f}s", flush=True)
