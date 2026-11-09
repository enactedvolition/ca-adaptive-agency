"""Experiment 9a: rerun the 7 Exp 5 variants x seeds 0-2 to step 2500 with sim7 (sim5 + per-agent lifetime
deposit attribution, observer-only). Same harness as exp5_price.py (3 seeds vmapped per variant, PRNGKey(1000*s+7),
10-step chunks) so the dynamics should reproduce the Exp 5 forks in runs5/ bit for bit.

Checks written to runs9/attr_{variant}.json:
  bit_vs_exp5_fork : every sim5 state field at t=2500 equals runs5/fork_{v}_s{seed}.npz exactly (except `anc`,
                     which the Exp 5 Price harness rewrote from Python every 10 steps; it feeds nothing back)
  attr_residual    : max |Adep.sum(0) + Aorph - Wl| / max(Wl)  (accounting identity, float32 order only)
Output per run: runs9/fork9_{v}_s{seed}.npz (full state incl. Adep, Aorph; NOT committed — large).
  python exp9_attr.py --procs 4 ;  python exp9_attr.py --bitcheck   # sim7(attr off/on) vs sim5, 300 steps
"""
from __future__ import annotations
import argparse, os, sys, time, json
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP9_OUT", os.path.join(HERE, "runs9"))
FORKS5 = os.path.join(HERE, "runs5")
FORK_T = 2500; DT = 10
SEEDS = [0, 1, 2]

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def make(simmod, v, seeds, **cfgkw):
    import jax, jax.numpy as jnp, exp5_price as E5
    cfg = simmod.Cfg(N=2048, L=1.0, G=64, **cfgkw)
    binit, bchunk, bcol, consts = simmod.batched(cfg)
    prm = simmod.stack_params([simmod.params(**dict(E5.BASE, **E5.VARIANTS[v])) for _ in seeds])
    keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for s in seeds])
    return cfg, binit(keys, prm), bchunk, prm

def worker(v):
    setup()
    import jax, jax.numpy as jnp, sim7
    jpath = os.path.join(OUT, f"attr_{v}.json")
    if os.path.exists(jpath) and all(os.path.exists(os.path.join(OUT, f"fork9_{v}_s{s}.npz")) for s in SEEDS):
        return v, 0.0, "cached"
    t0 = time.time()
    cfg, st, bchunk, prm = make(sim7, v, SEEDS, attr=True)
    for t in range(DT, FORK_T + 1, DT):
        st = bchunk(st, prm, DT)
        if t % 500 == 0:
            print(v, t, f"{time.time()-t0:.0f}s", flush=True)
    res = dict(variant=v, t=FORK_T, seeds=SEEDS, per_seed={}, wall_s=None)
    for b, s in enumerate(SEEDS):
        d = {k: np.asarray(x[b]) for k, x in st.items()}
        z5 = np.load(os.path.join(FORKS5, f"fork_{v}_s{s}.npz"))
        mism = []
        for k in z5.files:
            if k == "anc": continue
            a5 = z5[k]; a7 = d[k]
            same = a5.shape == a7.shape and a5.dtype == a7.dtype and np.array_equal(a5, a7, equal_nan=True)
            if not same: mism.append(k)
        Wl = d["Wl"].astype(np.float64); acc = d["Adep"].astype(np.float64).sum(0) + d["Aorph"].astype(np.float64)
        resid = float(np.abs(acc - Wl).max() / max(Wl.max(), 1e-12))
        alive = d["alive"].astype(bool)
        res["per_seed"][s] = dict(bit_vs_exp5_fork=(len(mism) == 0), mismatched=mism, attr_residual=resid,
                                  wall_mass=float(Wl.sum()), live_attr_mass=float(d["Adep"].sum()), orphan_mass=float(d["Aorph"].sum()),
                                  n_alive=int(alive.sum()), dead_slots_nonzero=float(np.abs(d["Adep"][~alive]).max()) if (~alive).any() else 0.0)
        tmp = os.path.join(OUT, f"fork9_{v}_s{s}.tmp.npz")
        np.savez_compressed(tmp, **d); os.replace(tmp, os.path.join(OUT, f"fork9_{v}_s{s}.npz"))
    res["wall_s"] = time.time() - t0
    json.dump(res, open(jpath + ".tmp", "w"), indent=1); os.replace(jpath + ".tmp", jpath)
    return v, res["wall_s"], "ok " + json.dumps({s: (r["bit_vs_exp5_fork"], r["attr_residual"]) for s, r in res["per_seed"].items()})

def bitcheck(steps=300):
    """sim7 attr=False and attr=True vs sim5 in the same harness (main arm, 3 seeds)."""
    setup()
    import sim5, sim7
    v = "sw_pecost_pe"
    _, s5, c5, p5 = make(sim5, v, SEEDS)
    _, s7a, c7a, p7a = make(sim7, v, SEEDS, attr=False)
    _, s7b, c7b, p7b = make(sim7, v, SEEDS, attr=True)
    for _ in range(steps // DT):
        s5 = c5(s5, p5, DT); s7a = c7a(s7a, p7a, DT); s7b = c7b(s7b, p7b, DT)
    out = {}
    for name, s7 in (("attr_off", s7a), ("attr_on", s7b)):
        mism = [k for k in s5 if not np.array_equal(np.asarray(s5[k]), np.asarray(s7[k]), equal_nan=True)]
        out[name] = dict(n_fields=len(s5), mismatched=mism, bit_for_bit=len(mism) == 0)
    Wl = np.asarray(s7b["Wl"], np.float64); acc = np.asarray(s7b["Adep"], np.float64).sum(1) + np.asarray(s7b["Aorph"], np.float64)
    out["attr_residual"] = float(np.abs(acc - Wl).max() / Wl.max()); out["steps"] = steps; out["variant"] = v
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "bitcheck.json"), "w"), indent=1)
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--bitcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--only", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.bitcheck:
        bitcheck(a.steps); sys.exit(0)
    sys.path.insert(0, HERE)
    import exp5_price as E5
    V = [a.only] if a.only else list(E5.VARIANTS)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for v, w, d in pool.imap_unordered(worker, V):
            print("DONE", v, f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
