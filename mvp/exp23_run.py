"""Experiment 23: decompose A = r_CL - r_OPEN by action channel (pre-registered in NIGHTLY_LOG 2026-11-25). Exp 22 births-off assay plus partial-OPEN
arms replaying only a channel subset (sim13 ctl["amask"]): MOVE (a0,a1), EAT (a2), SOC (a4..a7), MOVEEAT (a0..a2); the rest closed-loop; rep (a3) off in all arms.
Reruns cl and open (must equal Exp 22 pickles bit-for-bit) and clamp0 of each partial arm (must equal ref).  python exp23_run.py --procs 4 [--pops true,ctrl,sham]
(resumable: one pickle per (pop, seed) in runs23/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs23"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp20_run as X20
T, K_T, DROP, E_MIN = X20.T, X20.K_T, X20.DROP, X20.E_MIN
OUT22 = os.path.join(HERE, "runs22")
MASKS = dict(full=[1]*8, move=[1,1,0,0,0,0,0,0], eat=[0,0,1,0,0,0,0,0], soc=[0,0,0,0,1,1,1,1], moveeat=[1,1,1,0,0,0,0,0])

def build(pop, seed, rep_off=True):
    """exp20_run.build / exp21_run.build plus the targets' rep-off override (rep_off=False -> exactly the Exp 20/21 ctl)."""
    import jax, jax.numpy as jnp, exp19_run as X, exp21_run as X21
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    if pop == "true":
        z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X20.pick_targets(base, off, seed)
    elif pop == "sham":   # optional age-matched check: Exp 19 SHAM rerun, state at t=1500 (runs22/sham_forks), any-agent targets (implant extinct)
        z = np.load(os.path.join(OUT22, "sham_forks", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X21.pick_targets(base, seed)
    elif pop == "ctrl":
        base = X.load_fork(seed, tmpl); tg = X21.pick_targets(base, seed)
    else: raise ValueError(pop)
    K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, drop, clamp, frozen, aov, eref, am):
        s = dict(s); s["key"] = key; uid0 = s["uid"][tg_j]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        def body(c, x):
            a_t, e_t = x; same = c["uid"][tg_j] == uid0; pre_t = c["t"]
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(clamp & same & (jnp.abs(a_t).sum(1) > 0)), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       amask=jnp.zeros((N, 8), bool).at[tg_j].set(jnp.broadcast_to(am, (K, 8))),
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
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run23_{pop}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(pop, seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    ALL = jnp.ones(8, bool); M = {k: jnp.asarray(v, bool) for k, v in MASKS.items()}
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    p22 = pickle.load(open(os.path.join(OUT22, f"run22_{pop}_s{seed}.pkl"), "rb"))
    res = dict(pop=pop, seed=seed, targets=tg, E0=base["E"][tg], K=K, T=T, arms={})
    ref = g(f(bj, key0, 0.0, F, F, z8, ze, ALL)); aref = jnp.asarray(ref["act"])
    res["ref_matches_exp22"] = bool(np.array_equal(ref["E"], p22["arms"]["ref"]["E"]) and np.array_equal(ref["passed"], p22["arms"]["ref"]["passed"]) and np.array_equal(tg, p22["targets"]))
    res["arms"]["ref"] = {k: v for k, v in ref.items() if k != "act"}
    res["arms"]["cl"] = g(f(bj, key0, 1.0, F, F, z8, ze, ALL))
    for nm, m in M.items():
        res["arms"][nm] = g(f(bj, key0, 1.0, Tr, F, aref, ze, m))
        c0 = g(f(bj, key0, 0.0, Tr, F, aref, ze, m)); res["arms"]["clamp0_" + nm] = c0
    res["match22"] = {a: bool(np.array_equal(res["arms"][b_]["E"], p22["arms"][a]["E"]) and np.array_equal(res["arms"][b_]["passed"], p22["arms"][a]["passed"]))
                      for a, b_ in (("cl", "cl"), ("open", "full"))}
    res["clamp0_maxdiff_E"] = {nm: float(np.abs(res["arms"]["clamp0_" + nm]["E"] - ref["E"]).max()) for nm in M}
    res["births_total"] = {a: int(v["births"].sum()) for a, v in res["arms"].items()}
    for a in res["arms"].values(): a.pop("act", None)
    res["twin"] = dict(E=p22["arms"]["twin"]["E"], passed=p22["arms"]["twin"]["passed"])   # passive noise reference, reused from Exp 22 (same twin key)
    res["wall_s"] = time.time() - t0
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); ap.add_argument("--pops", default="true,ctrl,sham"); a = ap.parse_args()
    jobs = [(p, int(s)) for p in a.pops.split(",") for s in a.seeds.split(",")]; t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w in pool.imap_unordered(run_seed, jobs): print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
