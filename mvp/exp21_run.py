"""Experiment 21: the Exp 20 perturbation-recovery assay on a NO-IMPLANT control population (pre-registered in NIGHTLY_LOG 2026-11-23).
Forks: runs17/fork16_sw_pays_s{seed}_t3000.npz (Exp 17 sw_pays, blind low-rate brains; = Exp 19 t=0, before the implant), loaded with
exp19_run.load_fork. Targets: 40 random live agents with E0 >= 0.3 (rng 2020 + seed; no implant marker). Everything else is exp20_run:
same arms (ref, twin, cl, open, frozen, clamp0), drop 50%, T=150, same recovery metric. No sim change.
  python exp21_run.py --procs 4   (resumable: one pickle per seed in runs21/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs21"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp20_run as X20
T, K_T, DROP, E_MIN = X20.T, X20.K_T, X20.DROP, X20.E_MIN
ARMS = X20.ARMS; lineage_L = X20.lineage_L; recovery = X20.recovery

def pick_targets(base, seed, k=K_T):
    c = np.nonzero(base["alive"] & (base["E"] >= E_MIN))[0]; rng = np.random.default_rng(2020 + seed)
    return np.sort(rng.choice(c, size=min(k, len(c)), replace=False))

def build(seed):
    """identical to exp20_run.build except the fork (Exp 17 sw_pays t=3000) and the target rule (no marker)."""
    import jax, jax.numpy as jnp, exp19_run as X
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    base = X.load_fork(seed, tmpl)
    tg = pick_targets(base, seed); K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, drop, clamp, frozen, aov, eref):
        s = dict(s); s["key"] = key; uid0 = s["uid"][tg_j]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        def body(c, x):
            a_t, e_t = x; same = c["uid"][tg_j] == uid0; pre_t = c["t"]
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(clamp & same & (jnp.abs(a_t).sum(1) > 0)), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       e_off=jnp.zeros(N).at[tg_j].set(jnp.where(frozen & same & (e_t >= 0), e_t - c["E"][tg_j], 0.0)))
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0)
            ch = c2["alive"][None, :] & (c2["parent"][None, :] == uid0[:, None]) & (c2["birth"][None, :] == pre_t)
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], passed=(ch * c2["E"][None, :]).sum(1), births=ch.sum(1))
        _, rec = jax.lax.scan(body, s, (aov, eref)); return rec
    return jax.jit(run_one), base, tg, K, off

def run_seed(seed):
    import jax, jax.numpy as jnp
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run21_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    res = dict(seed=seed, targets=tg, E0=base["E"][tg], uid0=base["uid"][tg], pop0=int(base["alive"].sum()), Elive0=base["E"][base["alive"]], K=K, T=T, arms={})
    res["arms"]["ref"] = ref = g(f(bj, key0, 0.0, F, F, z8, ze))
    eref = jnp.asarray(np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0), jnp.float32); aref = jnp.asarray(ref["act"])   # eref -1 = ref target already dead
    res["arms"]["twin"] = g(f(bj, jax.random.fold_in(key0, 777), 0.0, F, F, z8, ze))
    res["arms"]["cl"] = g(f(bj, key0, 1.0, F, F, z8, ze))
    res["arms"]["open"] = g(f(bj, key0, 1.0, Tr, F, aref, ze))
    res["arms"]["frozen"] = g(f(bj, key0, 1.0, F, Tr, z8, eref))
    res["arms"]["clamp0"] = g(f(bj, key0, 0.0, Tr, F, aref, ze))
    res["clamp0_maxdiff_E"] = float(np.abs(res["arms"]["clamp0"]["E"] - ref["E"]).max()); res["clamp0_maxdiff_P"] = float(np.abs(res["arms"]["clamp0"]["passed"] - ref["passed"]).max())
    res["ref_rep_rate"] = float((ref["act"][..., 3][ref["alive"]] > 0).mean())   # targets' rep-action > 0 fraction while alive (ref)
    for a in res["arms"].values(): a.pop("act", None) if a is not ref else None
    res["wall_s"] = time.time() - t0
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

def _w(sd): return run_seed(sd)
if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); a = ap.parse_args()
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w in pool.imap_unordered(_w, [int(x) for x in a.seeds.split(",")]): print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
