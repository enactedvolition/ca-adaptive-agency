"""Experiment 24: state-matched passive control for movement (pre-registered in NIGHTLY_LOG 2026-11-26). Exp 22 births-off assay plus a gradient-conditioned
open loop (GC): the target's thrust = tanh(map . current sensed resource gradient), map fitted per target on its own ref run (energy-blind), other channels
closed-loop (sim13 ctl["tuse"]/ctl["tmap"]). GC0 = same without the drop (baseline of GC). Secondary GCX/GCX0: map on all non-energy senses.
Reruns cl/open/frozen (must equal Exp 22 pickles bit-for-bit), move (must equal Exp 23), clamp0 (must equal ref).  python exp24_run.py --procs 4 [--pops true,ctrl,sham]
(resumable: one pickle per (pop, seed) in runs24/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs24"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp20_run as X20
T, K_T, DROP, E_MIN = X20.T, X20.K_T, X20.DROP, X20.E_MIN
OUT22, OUT23 = os.path.join(HERE, "runs22"), os.path.join(HERE, "runs23")
NX = 16; BIAS = 14
COLS = dict(gc=[1, 2, BIAS], gcx=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, BIAS])   # gc: sensed resource gradient (+bias); gcx: all non-energy senses
RIDGE = dict(gc=0.0, gcx=1e-2); MIN_PTS = 20; CLIPT = 0.995

def build(pop, seed, rep_off=True):
    """exp23_run.build plus the Exp 24 thrust-map hook (tuse False -> the Exp 23 ctl) and the sensed inputs (xprev) recorded for the targets."""
    import jax, jax.numpy as jnp, exp19_run as X, exp21_run as X21
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    if pop == "true":
        z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X20.pick_targets(base, off, seed)
    elif pop == "sham":
        z = np.load(os.path.join(OUT22, "sham_forks", f"fork20_s{seed}_t{X20.TF}.npz"))
        base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}; tg = X21.pick_targets(base, seed)
    elif pop == "ctrl":
        base = X.load_fork(seed, tmpl); tg = X21.pick_targets(base, seed)
    else: raise ValueError(pop)
    K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, drop, clamp, frozen, aov, eref, am, tu, tmap):
        s = dict(s); s["key"] = key; uid0 = s["uid"][tg_j]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        TM = jnp.zeros((N, NX, 2)).at[tg_j].set(tmap)
        def body(c, x):
            a_t, e_t = x; same = c["uid"][tg_j] == uid0; pre_t = c["t"]
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(clamp & same & (jnp.abs(a_t).sum(1) > 0)), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       amask=jnp.zeros((N, 8), bool).at[tg_j].set(jnp.broadcast_to(am, (K, 8))),
                       e_off=jnp.zeros(N).at[tg_j].set(jnp.where(frozen & same & (e_t >= 0), e_t - c["E"][tg_j], 0.0)),
                       tuse=jnp.zeros(N, bool).at[tg_j].set(tu & same), tmap=TM)
            if rep_off: ctl.update(rep_use=jnp.zeros(N, bool).at[tg_j].set(same), rep_over=jnp.full((N,), -1.0))
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0)
            ch = c2["alive"][None, :] & (c2["parent"][None, :] == uid0[:, None]) & (c2["birth"][None, :] == pre_t)
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], xs=c2["xprev"][tg_j], passed=(ch * c2["E"][None, :]).sum(1),
                            births=ch.sum(1), pop=c2["alive"].sum())
        _, rec = jax.lax.scan(body, s, (aov, eref)); return rec
    return jax.jit(run_one), base, tg, K, off

def fit_maps(act, xs, alive, kind):
    """Per-target least squares of arctanh(clip(thrust)) on the chosen sensed channels over the ref target's live steps; < MIN_PTS -> pooled map.
    Returns tmap (K,16,2) float32, per-target R^2 of tanh(fit) vs thrust (fit steps), n points, pooled flag."""
    cols = COLS[kind]; lam = RIDGE[kind]; K = act.shape[1]
    def solve(Xm, Y):
        A = Xm.T @ Xm; reg = lam * len(Xm) * np.eye(len(cols)); reg[cols.index(BIAS), cols.index(BIAS)] = 0.0
        return np.linalg.solve(A + reg + 1e-9 * np.eye(len(cols)), Xm.T @ Y)
    th = np.arctanh(np.clip(act[..., :2].astype(np.float64), -CLIPT, CLIPT)); Xa = xs[..., cols].astype(np.float64)
    m = alive.astype(bool); Wp = solve(Xa[m], th[m])
    tmap = np.zeros((K, NX, 2)); r2 = np.full(K, np.nan); npts = m.sum(0); pooled = npts < MIN_PTS
    for k in range(K):
        mk = m[:, k]; W = Wp if pooled[k] else solve(Xa[mk, k], th[mk, k]); tmap[k, cols] = W
        if mk.sum() >= 2:
            y = act[mk, k, :2].astype(np.float64); yh = np.tanh(Xa[mk, k] @ W)
            r2[k] = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean(0)) ** 2).sum(), 1e-12)
    return tmap.astype(np.float32), r2, npts, pooled

def run_seed(job):
    pop, seed = job
    import jax, jax.numpy as jnp
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run24_{pop}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base, tg, K, off = build(pop, seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((T, K, 8)); ze = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    ALL = jnp.ones(8, bool); MOVE = jnp.asarray([1, 1, 0, 0, 0, 0, 0, 0], bool); tz = jnp.zeros((K, NX, 2))
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    p22 = pickle.load(open(os.path.join(OUT22, f"run22_{pop}_s{seed}.pkl"), "rb")); p23 = pickle.load(open(os.path.join(OUT23, f"run23_{pop}_s{seed}.pkl"), "rb"))
    res = dict(pop=pop, seed=seed, targets=tg, E0=base["E"][tg], K=K, T=T, arms={}, fit={})
    ref = g(f(bj, key0, 0.0, F, F, z8, ze, ALL, F, tz)); aref = jnp.asarray(ref["act"])
    eref = jnp.asarray(np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0), jnp.float32)
    res["ref_matches_exp22"] = bool(np.array_equal(ref["E"], p22["arms"]["ref"]["E"]) and np.array_equal(ref["passed"], p22["arms"]["ref"]["passed"]) and np.array_equal(tg, p22["targets"]))
    A = res["arms"]; A["ref"] = ref
    A["cl"] = g(f(bj, key0, 1.0, F, F, z8, ze, ALL, F, tz))
    A["open"] = g(f(bj, key0, 1.0, Tr, F, aref, ze, ALL, F, tz))
    A["frozen"] = g(f(bj, key0, 1.0, F, Tr, z8, eref, ALL, F, tz))
    A["move"] = g(f(bj, key0, 1.0, Tr, F, aref, ze, MOVE, F, tz))
    A["clamp0"] = g(f(bj, key0, 0.0, Tr, F, aref, ze, ALL, F, tz))
    for kind in ("gc", "gcx"):
        tmap, r2, npts, pooled = fit_maps(ref["act"], ref["xs"], ref["alive"], kind)
        res["fit"][kind] = dict(tmap=tmap, r2=r2, npts=npts, pooled=pooled); tm = jnp.asarray(tmap)
        A[kind] = g(f(bj, key0, 1.0, F, F, z8, ze, ALL, Tr, tm)); A[kind + "0"] = g(f(bj, key0, 0.0, F, F, z8, ze, ALL, Tr, tm))
        # hook check on real data: logged thrust of live GC0 targets equals tanh(xs . tmap)
        a0 = A[kind + "0"]; lv = a0["alive"]; pred = np.tanh(np.einsum("tki,kic->tkc", a0["xs"].astype(np.float32), tmap))
        res["fit"][kind]["hook_maxdiff"] = float(np.abs(pred[lv] - a0["act"][lv][:, :2]).max()) if lv.any() else 0.0
    eq = lambda a, b: bool(np.array_equal(a["E"], b["E"]) and np.array_equal(a["passed"], b["passed"]))
    res["match22"] = {a: eq(A[a], p22["arms"][a]) for a in ("cl", "open", "frozen")}
    res["match23"] = dict(move=eq(A["move"], p23["arms"]["move"]))
    res["clamp0_maxdiff_E"] = float(np.abs(A["clamp0"]["E"] - ref["E"]).max())
    res["gc0_maxdiff_E"] = {k: float(np.abs(A[k + "0"]["E"] - ref["E"]).max()) for k in ("gc", "gcx")}
    res["births_total"] = {a: int(v["births"].sum()) for a, v in A.items()}
    for a in A.values(): a.pop("act", None); a.pop("xs", None)
    res["twin"] = dict(E=p22["arms"]["twin"]["E"], passed=p22["arms"]["twin"]["passed"])
    res["wall_s"] = time.time() - t0
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); ap.add_argument("--pops", default="true,ctrl,sham"); a = ap.parse_args()
    jobs = [(p, int(s)) for p in a.pops.split(",") for s in a.seeds.split(",")]; t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w in pool.imap_unordered(run_seed, jobs): print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
