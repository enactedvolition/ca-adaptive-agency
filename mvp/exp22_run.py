"""Experiment 22: FORAGING recovery with births disabled by construction (pre-registered in NIGHTLY_LOG 2026-11-24).
The Exp 20/21 energy-drop assay (40 targets, 50% E drop, T=150; arms ref, twin, cl, open, frozen, clamp0), with the TARGETS' rep action forced
to -1 (no birth attempt) in every arm while the slot holds the target's uid, via sim13's existing observer hook ctl["rep_use"]/ctl["rep_over"]
(applied after the OPEN action clamp). Rest of the world unchanged. Populations: "true" = Exp 20 TRUE t=1500 forks (runs20/, marker targets,
exp20_run.pick_targets); "ctrl" = no-implant Exp 17 sw_pays t=3000 forks (runs17/, exp21_run.pick_targets). Scored on own E (analyze22.py).
No sim change.   python exp22_run.py --procs 4 [--pops true,ctrl] [--seeds 0,...,8]   (resumable: one pickle per (pop, seed) in runs22/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs22"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp20_run as X20
T, K_T, DROP, E_MIN = X20.T, X20.K_T, X20.DROP, X20.E_MIN
ARMS = X20.ARMS

def build(pop, seed, rep_off=True):
    """exp20_run.build / exp21_run.build plus the targets' rep-off override (rep_off=False -> exactly the Exp 20/21 ctl)."""
    import jax, jax.numpy as jnp, exp19_run as X, exp21_run as X21
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    if pop == "true":
        z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X20.pick_targets(base, off, seed)
    elif pop == "sham":   # optional age-matched check: Exp 19 SHAM rerun, state at t=1500 (runs22/sham_forks), any-agent targets (implant extinct)
        z = np.load(os.path.join(OUT, "sham_forks", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X21.pick_targets(base, seed)
    elif pop == "ctrl":
        base = X.load_fork(seed, tmpl); tg = X21.pick_targets(base, seed)
    else: raise ValueError(pop)
    K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, drop, clamp, frozen, aov, eref):
        s = dict(s); s["key"] = key; uid0 = s["uid"][tg_j]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        def body(c, x):
            a_t, e_t = x; same = c["uid"][tg_j] == uid0; pre_t = c["t"]
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(clamp & same & (jnp.abs(a_t).sum(1) > 0)), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       e_off=jnp.zeros(N).at[tg_j].set(jnp.where(frozen & same & (e_t >= 0), e_t - c["E"][tg_j], 0.0)))
            if rep_off: ctl.update(rep_use=jnp.zeros(N, bool).at[tg_j].set(same), rep_over=jnp.full((N,), -1.0))
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0)
            ch = c2["alive"][None, :] & (c2["parent"][None, :] == uid0[:, None]) & (c2["birth"][None, :] == pre_t)
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], passed=(ch * c2["E"][None, :]).sum(1), births=ch.sum(1),
                            pop=c2["alive"].sum())
        _, rec = jax.lax.scan(body, s, (aov, eref)); return rec
    return jax.jit(run_one), base, tg, K, off

def run_seed(job):
    pop, seed = job
    import jax, jax.numpy as jnp
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run22_{pop}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(pop, seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    res = dict(pop=pop, seed=seed, targets=tg, E0=base["E"][tg], uid0=base["uid"][tg], pop0=int(base["alive"].sum()), Elive0=base["E"][base["alive"]], K=K, T=T, arms={})
    res["arms"]["ref"] = ref = g(f(bj, key0, 0.0, F, F, z8, ze))
    eref = jnp.asarray(np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0), jnp.float32); aref = jnp.asarray(ref["act"])
    res["arms"]["twin"] = g(f(bj, jax.random.fold_in(key0, 777), 0.0, F, F, z8, ze))
    res["arms"]["cl"] = g(f(bj, key0, 1.0, F, F, z8, ze))
    res["arms"]["open"] = g(f(bj, key0, 1.0, Tr, F, aref, ze))
    res["arms"]["frozen"] = g(f(bj, key0, 1.0, F, Tr, z8, eref))
    res["arms"]["clamp0"] = g(f(bj, key0, 0.0, Tr, F, aref, ze))
    res["clamp0_maxdiff_E"] = float(np.abs(res["arms"]["clamp0"]["E"] - ref["E"]).max())
    res["births_total"] = {a: int(v["births"].sum()) for a, v in res["arms"].items()}
    res["ref_rep_rate"] = float((ref["act"][..., 3][ref["alive"]] > 0).mean())   # brain's rep-action > 0 fraction while alive (suppressed)
    for nm, a in res["arms"].items(): res[f"{nm}_rep_rate"] = float((a["act"][..., 3][a["alive"]] > 0).mean()) if a["alive"].any() else float("nan")
    for a in res["arms"].values(): a.pop("act", None) if a is not ref else None
    res["wall_s"] = time.time() - t0
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); ap.add_argument("--pops", default="true,ctrl"); a = ap.parse_args()
    jobs = [(p, int(s)) for p in a.pops.split(",") for s in a.seeds.split(",")]; t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w in pool.imap_unordered(run_seed, jobs): print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
