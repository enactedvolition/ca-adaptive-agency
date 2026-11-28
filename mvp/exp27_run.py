"""Experiment 27 (pre-registered in NIGHTLY_LOG 2026-11-28): validate the Exp 26 drop assay on systems whose energy feedback is KNOWN.
World: Exp 20 TRUE t=1500 forks, sim13 rep_fixed world, births off for all agents, predation haz (default 0.004), brains drive the
8 non-eat actions with their energy sense clamped exactly (sim13 traced param blind=1), and a controller under test sets eat for every
cohort agent. So the controller is the only route from own energy to action (ground truth by construction).
Controller spec (37 floats): [kind, g, b, src, w(33)]; kind 0 P/const: eat=clip(b+g(0.40-u),0,1); 1 STEP: eat=1[u<0.40];
2 MLP: eat=sigmoid(w2.tanh(W1.[log1p(u)-0.7, R_here]+b1)+b2). src 0 own E, 1 E of a random other live cohort agent (sham), 2 const 0.40.
All constructed systems are POSITIVE/NEGATIVE CONTROLS, not evidence about evolved or real systems.
  python exp27_run.py assay --procs 4 [--systems ...] [--reps 0,1,2,3,4] [--haz 0.004]
  python exp27_run.py es --name es|esblind          (ES training, checkpoint per generation in runs27/)
  python exp27_run.py pay --procs 4                  (input-ablation pay-check)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs27"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
T, DROP, E_MIN, HAZ, THETA, B0, TF, K_T = 150, 0.5, 0.3, 0.004, 0.40, 0.5, 1500, 200
NW = 33; NSPEC = 4 + NW; TPAY = 300
GAINS = [0.25, 0.5, 1, 2, 4, 8, 16]
SIGHTED = os.environ.get("EXP27_SIGHTED") == "1"   # POST HOC diagnostic only: brains keep their energy sense (blind=0), as in Exp 26
REPS = [0, 1, 2, 3, 4]; SEEDS = list(range(9))

def w_init():
    w = np.zeros(NW, np.float32); w[:16] = np.random.default_rng(27).normal(0, 0.5, 16); return w

def spec(kind, g=0.0, b=B0, src=0, w=None):
    v = np.zeros(NSPEC, np.float32); v[:4] = (kind, g, b, src)
    if w is not None: v[4:] = w
    return v

def es_theta(name):
    p = os.path.join(OUT, f"es_{name}.npz")
    return np.load(p)["theta"].astype(np.float32) if os.path.exists(p) else None

def systems():
    S = {f"P{g}": spec(0, g) for g in GAINS}
    S.update({"P0": spec(0, 0.0), "CONST0.8": spec(0, 0.0, 0.8), "STEP": spec(1), "SHAM_STEP": spec(1, src=1), "SHAM_P4": spec(0, 4.0, src=1)})
    te, tb = es_theta("es"), es_theta("esblind")
    if te is not None: S["ES"] = spec(2, w=te); S["ES_ABL"] = spec(2, src=2, w=te)
    if tb is not None: S["ES_BLIND"] = spec(2, src=2, w=tb)
    return S
BLIND = ["P0", "CONST0.8", "SHAM_STEP", "SHAM_P4", "ES_BLIND", "ES_ABL"]
IGNORES_INPUT = ["P0", "CONST0.8", "ES_BLIND", "ES_ABL"]

def policy_np(sp, u, rh):
    """numpy mirror of the in-sim controller (for G_eff and tests)."""
    u = np.asarray(u, np.float64); kind, g, b, src = sp[:4]
    if src == 2: u = np.full_like(u, THETA)
    if kind == 0: return np.clip(b + g * (THETA - u), 0, 1)
    if kind == 1: return (u < THETA).astype(float)
    w = sp[4:].astype(np.float64); x = np.stack([np.log1p(np.maximum(u, 0)) - 0.7, np.asarray(rh, np.float64) * np.ones_like(u)], -1)
    h = np.tanh(x @ w[:16].reshape(2, 8) + w[16:24]); return 1 / (1 + np.exp(-(h @ w[24:32] + w[32])))

_C = {}
def ctx():
    if "w" in _C: return _C
    import jax, jax.numpy as jnp, exp19_run as X
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); prm = dict(prm); prm["blind"] = jnp.float32(0.0 if SIGHTED else 1.0)
    step = consts["step"]; N = cfg.N; G = cfg.G; cs = cfg.L / G
    def ctrl(sp, u, rh):      # sp (N, NSPEC)
        kind, g, b, src = sp[:, 0], sp[:, 1], sp[:, 2], sp[:, 3]; w = sp[:, 4:]
        p = jnp.clip(b + g * (THETA - u), 0.0, 1.0); st = (u < THETA).astype(jnp.float32)
        x = jnp.stack([jnp.log1p(jnp.maximum(u, 0.0)) - 0.7, rh], 1)
        h = jnp.tanh(jnp.einsum("ni,nih->nh", x, w[:, :16].reshape(-1, 2, 8)) + w[:, 16:24])
        m = jax.nn.sigmoid((h * w[:, 24:32]).sum(1) + w[:, 32])
        return jnp.where(kind == 0, p, jnp.where(kind == 1, st, m))
    def inputs(c, sp, same, k1):
        E = c["E"]; lc = same.astype(jnp.float32); idx = jax.random.choice(k1, N, (N,), p=lc / jnp.maximum(lc.sum(), 1.0))
        u = jnp.where(sp[:, 3] == 0, E, jnp.where(sp[:, 3] == 1, E[idx], jnp.float32(THETA)))
        cell = jnp.clip((c["pos"] / cs).astype(jnp.int32), 0, G - 1); rh = c["Rf"][cell[:, 0], cell[:, 1]]
        return u, rh
    def assay_one(s, kx, tg_j, haz, drop, clamp, frozen, aov, eov, uref, sp1):
        sp = jnp.broadcast_to(sp1, (N, NSPEC)); s = dict(s); coh = s["alive"]; uid0 = s["uid"]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        def body(c, x):
            k, a_t, ea_t, ur_t = x; k1, k2 = jax.random.split(k)
            same = c["alive"] & coh & (c["uid"] == uid0); sameT = same[tg_j]
            refal = jnp.abs(a_t).sum(1) > 0
            cl_T = clamp & sameT & refal; fr_T = frozen & sameT & (ur_t >= 0)
            u, rh = inputs(c, sp, same, k1)
            u = u.at[tg_j].set(jnp.where(fr_T, ur_t, u[tg_j]))
            ea = ctrl(sp, u, rh); eov_all = ea.at[tg_j].set(jnp.where(cl_T, ea_t, ea[tg_j]))
            hu = jax.random.uniform(k2, (N,))
            eo = jnp.zeros(N).at[tg_j].set(jnp.where(fr_T, ur_t - c["E"][tg_j], 0.0)) if SIGHTED else jnp.zeros(N)   # post hoc sighted mode: FROZEN also freezes the brains' sensed E (Exp 26 style)
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(cl_T), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t), e_off=eo,
                       rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0), eat_use=same, eat_over=eov_all, haz=haz, hu=hu)
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0[tg_j])
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], eat=eov_all[tg_j], u=u[tg_j],
                            births=c2["repstep"].sum(), pred=sameT & (hu[tg_j] < haz * eov_all[tg_j]))
        _, rec = jax.lax.scan(body, s, (jax.random.split(kx, T), aov, eov, uref)); return rec
    def group_run(s, kx, haz, sps, grp, ng):
        """sps (N, NSPEC) per-agent controllers; grp (N,) int group (cohort only); returns alive counts per group per step."""
        coh = s["alive"]; uid0 = s["uid"]
        def body(c, k):
            k1, k2 = jax.random.split(k)
            same = c["alive"] & coh & (c["uid"] == uid0)
            u, rh = inputs(c, sps, same, k1); ea = ctrl(sps, u, rh); hu = jax.random.uniform(k2, (N,))
            ctl = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0),
                       eat_use=same, eat_over=ea, haz=haz, hu=hu)
            c2 = step(c, prm, ctl); now = c2["alive"] & coh & (c2["uid"] == uid0)
            return c2, dict(alive=jax.ops.segment_sum(now.astype(jnp.int32), grp, ng), eat=jax.ops.segment_sum(jnp.where(same, ea, 0.0), grp, ng))
        _, rec = jax.lax.scan(body, s, jax.random.split(kx, TPAY)); return rec
    _C.update(jax=jax, jnp=jnp, tmpl=tmpl, N=N, G=G, cs=cs, assay=jax.jit(assay_one), group=jax.jit(group_run, static_argnums=5), w=True)
    return _C

def load_base(seed):
    C = ctx(); z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{TF}.npz"))
    return {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in C["tmpl"].items()}

def targets(base, seed, rep):
    elig = np.nonzero(base["alive"] & (base["E"] >= E_MIN))[0]; assert len(elig) >= K_T
    return np.sort(np.random.default_rng(27000 + 1000 * rep + seed).choice(elig, size=K_T, replace=False))

def assay_job(job, nsteps=None):
    name, sp, rep, seed, haz = job
    tag = f"{name}_h{haz:g}_r{rep}_s{seed}" + ("_sighted" if SIGHTED else ""); path = os.path.join(OUT, f"a27_{tag}.pkl")
    if nsteps is None and os.path.exists(path): return path, 0.0
    t0 = time.time(); C = ctx(); jnp = C["jnp"]; jax = C["jax"]
    base = load_base(seed); tg = targets(base, seed, rep); tg_j = jnp.asarray(tg, jnp.int32)
    bj = {k: jnp.asarray(v) for k, v in base.items()}; kx = jax.random.PRNGKey(27500 + 1000 * rep + seed); h = jnp.float32(haz); s1 = jnp.asarray(sp)
    z8 = jnp.zeros((T, K_T, 8)); zk = jnp.zeros((T, K_T)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    f = lambda d, cl, fr, a, e, ur: g(C["assay"](bj, kx, tg_j, h, d, cl, fr, a, e, ur, s1))
    ref = f(0.0, F, F, z8, zk, zk - 1)
    a_ref = jnp.asarray(np.where(ref["alive"][..., None], ref["act"], 0.0)); ea_ref = jnp.asarray(ref["eat"]); u_ref = jnp.asarray(np.where(ref["alive"], ref["u"], -1.0))
    arms = dict(ref=ref, cl=f(1.0, F, F, z8, zk, zk - 1), open=f(1.0, Tr, F, a_ref, ea_ref, zk - 1), frozen=f(1.0, F, Tr, z8, zk, u_ref), clamp0=f(0.0, Tr, F, a_ref, ea_ref, zk - 1))
    cl0 = np.clip((base["pos"][tg] / C["cs"]).astype(np.int32), 0, C["G"] - 1); rh0 = np.asarray(base["Rf"])[cl0[:, 0], cl0[:, 1]]
    E0 = base["E"][tg].astype(np.float64)
    geff = float(np.mean((policy_np(sp, 0.5 * E0, rh0) - policy_np(sp, E0, rh0)) / (0.5 * E0))) if sp[3] != 1 else float("nan")
    keep = {a: {k: v for k, v in arms[a].items() if k != "act"} for a in arms}
    res = dict(sighted=SIGHTED, name=name, spec=sp, rep=rep, seed=seed, haz=haz, tg=tg, E0=E0, G_eff=geff,
               clamp0_exact=bool(np.array_equal(arms["clamp0"]["E"], ref["E"]) and np.array_equal(arms["clamp0"]["alive"], ref["alive"])),
               frozen_eq_cl=bool(np.array_equal(arms["frozen"]["E"], arms["cl"]["E"]) and np.array_equal(arms["frozen"]["alive"], arms["cl"]["alive"])),
               births=int(sum(arms[a]["births"].sum() for a in arms)), arms=keep, secs=time.time() - t0)
    if nsteps is not None: return res
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path); return path, time.time() - t0

def es_train(name, gens=100, pairs=8, sigma=0.1, lr=0.03):
    C = ctx(); jnp = C["jnp"]; jax = C["jax"]; N = C["N"]; blind = name == "esblind"; ck = os.path.join(OUT, f"es_{name}_ck.npz")
    if os.path.exists(ck):
        z = np.load(ck); theta, m, v, g0, hist = z["theta"], z["m"], z["v"], int(z["gen"]) + 1, list(z["hist"])
    else:
        theta, m, v, g0, hist = w_init(), np.zeros(NW), np.zeros(NW), 0, []
    nc = 2 * pairs; b1, b2 = 0.9, 0.999
    for gen in range(g0, gens):
        t0 = time.time(); rng = np.random.default_rng(91000 + gen); eps = rng.normal(0, 1, (pairs, NW)); eps = np.concatenate([eps, -eps])
        cand = theta[None] + sigma * eps; fit = np.zeros(nc)
        for j, seed in enumerate((gen % 9, (gen + 4) % 9)):
            base = load_base(seed); coh = np.nonzero(base["alive"])[0]
            grp = np.full(N, nc, np.int32); grp[coh] = rng.permutation(np.arange(len(coh)) % nc)
            sps = np.zeros((N, NSPEC), np.float32); sps[:, 0] = 2; sps[:, 3] = 2 if blind else 0; sps[:, 4:] = np.concatenate([cand, theta[None]])[grp]
            rec = C["group"]({k: jnp.asarray(x) for k, x in base.items()}, jax.random.PRNGKey(90000 + 10 * gen + j), jnp.float32(HAZ), jnp.asarray(sps), jnp.asarray(grp), nc + 1)
            al = np.asarray(rec["alive"]); n0 = np.bincount(grp[coh], minlength=nc + 1)
            fit += al[-1, :nc] / n0[:nc] / 2
        rk = np.empty(nc); rk[np.argsort(fit)] = np.arange(nc); rk = rk / (nc - 1) - 0.5
        grad = (rk[:, None] * eps).sum(0) / (nc * sigma)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad**2; t = gen + 1
        theta = (theta + lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)).astype(np.float32)
        hist.append([gen, fit.mean(), fit.max(), time.time() - t0])
        np.savez(ck + ".tmp.npz", theta=theta, m=m, v=v, gen=gen, hist=np.array(hist)); os.replace(ck + ".tmp.npz", ck)
        print(time.strftime("%H:%M:%S"), name, gen, f"mean {fit.mean():.3f} max {fit.max():.3f} {time.time()-t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, f"es_{name}.npz"), theta=theta, hist=np.array(hist))

def pay_job(job):
    """within-world parity pay-check (Exp 25 design), T=300, keys PRNGKey(2500+seed): A on one uid parity, B on the other, both parities."""
    nA, spA, nB, spB, seed = job; path = os.path.join(OUT, f"p27_{nA}_vs_{nB}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); C = ctx(); jnp = C["jnp"]; jax = C["jax"]; N = C["N"]; base = load_base(seed); bj = {k: jnp.asarray(x) for k, x in base.items()}
    coh = base["alive"]; out = []
    for par in (0, 1):
        isA = coh & ((base["uid"] % 2) == par); grp = np.where(isA, 0, np.where(coh, 1, 2)).astype(np.int32)
        sps = np.where(isA[:, None], spA[None], spB[None]).astype(np.float32)
        rec = C["group"](bj, jax.random.PRNGKey(2500 + seed), jnp.float32(HAZ), jnp.asarray(sps), jnp.asarray(grp), 3)
        n0 = np.bincount(grp, minlength=3)[:2]; al = np.asarray(rec["alive"])[:, :2] / n0; out.append(al)
    al = (out[0] + out[1]) / 2
    pickle.dump(dict(A=nA, B=nB, seed=seed, surv=al, secs=time.time() - t0), open(path + ".tmp", "wb")); os.replace(path + ".tmp", path); return path, time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--systems", default="")
    ap.add_argument("--reps", default="0,1,2,3,4"); ap.add_argument("--haz", type=float, default=HAZ); ap.add_argument("--name", default="es"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); import multiprocessing as mp
    if a.mode == "es": es_train(a.name); sys.exit()
    if a.mode == "assay":
        S = systems(); names = a.systems.split(",") if a.systems else list(S)
        jobs = [(n, S[n], r, s, a.haz) for r in [int(x) for x in a.reps.split(",")] for n in names for s in SEEDS]
        fn = assay_job
    else:
        S = systems(); pr = [("ES", "ES_ABL"), ("ES", "P0"), ("ES", "STEP"), ("ES", "ES_BLIND"), ("STEP", "P0")]
        jobs = [(x, S[x], y, S[y], s) for x, y in pr for s in SEEDS]; fn = pay_job
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for path, sec in pool.imap_unordered(fn, jobs):
            print(time.strftime("%H:%M:%S"), os.path.basename(path), f"{sec:.0f}s", flush=True)
