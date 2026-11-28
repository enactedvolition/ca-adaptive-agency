"""Experiment 26 part 2 (pre-registered in NIGHTLY_LOG 2026-11-28): re-score the Exp 22-24 populations with the TARGET-SURVIVAL read-out, K = 200 targets/seed.
Exp 22/24 births-off drop assay (targets' rep forced off, no predation) with K=200 targets drawn by rng 2626+seed from alive & E>=0.3 (TRUE: implant-marked only).
Arms: ref, twin (key folded 777), cl, open, frozen, gc, gc0 (Exp 24 energy-blind gradient-conditioned thrust map fitted on the target's own ref run), clamp0.
Two configs, each T=300 scan (primary read at step 150 = prefix): d50 (50% drop, primary) and d80 (80% drop, secondary).
No sim change.   python exp26_pop.py --procs 4 [--pops true,ctrl,sham] [--seeds 0,...,8]   (resumable: one pickle per (pop, seed) in runs26/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs26"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp20_run as X20, exp24_run as X24
T, K_T, DROP, E_MIN = 300, 200, X20.DROP, X20.E_MIN
T1 = 150                                  # primary read-out step (survival after step T1)
NX = X24.NX
CONFIGS = {"d50": 1.0, "d80": 1.6}        # `drop` multiplier on DROP = 0.5

def pick_targets(pop, base, off, seed, k=K_T):
    import exp20_run
    ok = base["alive"] & (base["E"] >= E_MIN)
    if pop == "true": ok = ok & exp20_run.marker_np(base["W"], off)
    c = np.nonzero(ok)[0]; rng = np.random.default_rng(2626 + seed)
    return np.sort(rng.choice(c, size=min(k, len(c)), replace=False))

def survival(alive, t=T1):
    """alive (T, K) bool own-slot indicator after each step; survival fraction after step t (1-indexed)."""
    return float(np.asarray(alive)[t - 1].mean())

def load_base(pop, seed, tmpl):
    import exp19_run as X
    if pop == "true":
        z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{X20.TF}.npz"))
    elif pop == "sham":
        z = np.load(os.path.join(HERE, "runs22", "sham_forks", f"fork20_s{seed}_t{X20.TF}.npz"))
    elif pop == "ctrl":
        return X.load_fork(seed, tmpl)
    else: raise ValueError(pop)
    return {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}

def build(pop, seed, tg=None, rep_off=True):
    import jax, jax.numpy as jnp, exp19_run as X
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    base = load_base(pop, seed, tmpl)
    if tg is None: tg = pick_targets(pop, base, off, seed)
    K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, drop, clamp, frozen, aov, eref, tu, tmap, nsteps=T):
        s = dict(s); s["key"] = key; uid0 = s["uid"][tg_j]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        TM = jnp.zeros((N, NX, 2)).at[tg_j].set(tmap)
        def body(c, x):
            a_t, e_t = x; same = c["uid"][tg_j] == uid0; pre_t = c["t"]
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(clamp & same & (jnp.abs(a_t).sum(1) > 0)), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       e_off=jnp.zeros(N).at[tg_j].set(jnp.where(frozen & same & (e_t >= 0), e_t - c["E"][tg_j], 0.0)),
                       tuse=jnp.zeros(N, bool).at[tg_j].set(tu & same), tmap=TM)
            if rep_off: ctl.update(rep_use=jnp.zeros(N, bool).at[tg_j].set(same), rep_over=jnp.full((N,), -1.0))
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0)
            ch = c2["alive"][None, :] & (c2["parent"][None, :] == uid0[:, None]) & (c2["birth"][None, :] == pre_t)
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], xs=c2["xprev"][tg_j], births=ch.sum(1))
        _, rec = jax.lax.scan(body, s, (aov[:nsteps], eref[:nsteps])); return rec
    return jax.jit(run_one, static_argnames=("nsteps",)), base, tg, K, off

def run_seed(job):
    pop, seed = job
    import jax, jax.numpy as jnp
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run26_{pop}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(pop, seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True); tz = jnp.zeros((K, NX, 2))
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    res = dict(pop=pop, seed=seed, targets=tg, E0=base["E"][tg], K=K, T=T, T1=T1, n_eligible=int((base["alive"] & (base["E"] >= E_MIN)).sum()), cfg={}, fit={})
    ref = g(f(bj, key0, 0.0, F, F, z8, ze, F, tz)); aref = jnp.asarray(ref["act"])
    eref = jnp.asarray(np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0), jnp.float32)
    tmap, r2, npts, pooled = X24.fit_maps(ref["act"], ref["xs"], ref["alive"], "gc"); tm = jnp.asarray(tmap)
    res["fit"] = dict(r2=r2, npts=npts, pooled=pooled)
    twin = g(f(bj, jax.random.fold_in(key0, 777), 0.0, F, F, z8, ze, F, tz))
    A0 = dict(ref=ref, twin=twin, gc0=g(f(bj, key0, 0.0, F, F, z8, ze, Tr, tm)), clamp0=g(f(bj, key0, 0.0, Tr, F, aref, ze, F, tz)))
    res["clamp0_maxdiff_E"] = float(np.abs(A0["clamp0"]["E"] - ref["E"]).max()); res["clamp0_alive_equal"] = bool(np.array_equal(A0["clamp0"]["alive"], ref["alive"]))
    res["ref_alive_cohort_T1"] = survival(ref["alive"])
    for cn, dm in CONFIGS.items():
        arms = dict(A0)
        arms["cl"] = g(f(bj, key0, dm, F, F, z8, ze, F, tz)); arms["open"] = g(f(bj, key0, dm, Tr, F, aref, ze, F, tz))
        arms["frozen"] = g(f(bj, key0, dm, F, Tr, z8, eref, F, tz)); arms["gc"] = g(f(bj, key0, dm, F, F, z8, ze, Tr, tm))
        for a in arms.values(): a.pop("act", None); a.pop("xs", None)
        res["cfg"][cn] = dict(arms={a: dict(alive=v["alive"], E=v["E"]) for a, v in arms.items()}, births_total={a: int(v["births"].sum()) for a, v in arms.items()})
    res["wall_s"] = time.time() - t0
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); ap.add_argument("--pops", default="true,ctrl,sham"); a = ap.parse_args()
    jobs = [(p, int(s)) for p in a.pops.split(",") for s in a.seeds.split(",")]; t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w in pool.imap_unordered(run_seed, jobs): print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
