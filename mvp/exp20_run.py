"""Experiment 20: perturbation recovery on the Exp 19 TRUE population (pre-registered in NIGHTLY_LOG 2026-11-22).
Forks: runs20/fork20_s{seed}_t1500.npz (Exp 19 TRUE rerun, state at t=1500; made by exp19_run.py with EXP19_FORKDIR). Per seed one world run per arm
(ref, twin, cl, open, frozen, clamp0) of T=150 steps; 40 targets (random live m=1 agents with E0>=0.3) lose 50% of E at step 0 in the drop arms.
OPEN = targets' 8 actions clamped to ref's recorded actions while the ref target lives (recorded act is all-zero once dead: then the brain acts itself); FROZEN = sensed-E channel replaced by ref's pre-step E while the ref target lives (e_off); records per step:
own E, energy passed to children (child E at birth), births. No sim change.   python exp20_run.py --procs 4   (resumable: one pickle per seed)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs20"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
T = int(os.environ.get("EXP20_T", "150")); K_T = 40; DROP = 0.5; E_MIN = 0.3; TF = 1500
ARMS = ("ref", "twin", "cl", "open", "frozen", "clamp0")

def marker_np(W, off):
    from exp19_run import MARK
    oi, (I, H) = off["W_ih"][0], off["W_ih"][1]; oo, (_, A) = off["W_ho"][0], off["W_ho"][1]
    return (W[:, oi + 8 * H: oi + 9 * H] * W[:, oo + 3: oo + H * A: A]).max(1) > MARK

def pick_targets(base, off, seed, k=K_T):
    m = marker_np(base["W"], off) & base["alive"] & (base["E"] >= E_MIN)
    c = np.nonzero(m)[0]; rng = np.random.default_rng(2020 + seed)
    return np.sort(rng.choice(c, size=min(k, len(c)), replace=False))

def lineage_L(E_rec, passed):
    """lineage energy L(t) = own E (0 if dead) + cumulative energy passed to children. Arrays (T, K)."""
    return E_rec + np.cumsum(passed, axis=0)

def recovery(arm_E, arm_P, ref_E, ref_P, E0, lo=T // 2, offspring=True):
    """per-target r = 1 - mean_{t in (lo, T]} (L_ref - L_arm) / dE, dE = DROP*E0."""
    La = lineage_L(arm_E, arm_P) if offspring else arm_E; Lr = lineage_L(ref_E, ref_P) if offspring else ref_E
    return 1.0 - (Lr - La)[lo:].mean(0) / (DROP * E0)

def build(seed):
    import jax, jax.numpy as jnp, exp19_run as X
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    z = np.load(os.path.join(OUT, f"fork20_s{seed}_t{TF}.npz"))
    base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}
    tg = pick_targets(base, off, seed); K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
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
    path = os.path.join(OUT, f"run20_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    res = dict(seed=seed, targets=tg, E0=base["E"][tg], uid0=base["uid"][tg], pop0=int(base["alive"].sum()), K=K, T=T, arms={})
    res["arms"]["ref"] = ref = g(f(bj, key0, 0.0, F, F, z8, ze))
    eref = jnp.asarray(np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0), jnp.float32); aref = jnp.asarray(ref["act"])   # eref -1 = ref target already dead
    res["arms"]["twin"] = g(f(bj, jax.random.fold_in(key0, 777), 0.0, F, F, z8, ze))
    res["arms"]["cl"] = g(f(bj, key0, 1.0, F, F, z8, ze))
    res["arms"]["open"] = g(f(bj, key0, 1.0, Tr, F, aref, ze))
    res["arms"]["frozen"] = g(f(bj, key0, 1.0, F, Tr, z8, eref))
    res["arms"]["clamp0"] = g(f(bj, key0, 0.0, Tr, F, aref, ze))
    res["clamp0_maxdiff_E"] = float(np.abs(res["arms"]["clamp0"]["E"] - ref["E"]).max()); res["clamp0_maxdiff_P"] = float(np.abs(res["arms"]["clamp0"]["passed"] - ref["passed"]).max())
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
