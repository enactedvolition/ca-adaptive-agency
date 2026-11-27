"""Experiment 25: pay-check for an energy-dependent trade-off (starvation vs predation), pre-registered in NIGHTLY_LOG 2026-11-27.
World: Exp 20 TRUE t=1500 forks (sim13 rep_fixed world as Exp 19/20), births off for every agent (rep forced -1), sim13's new optional
predation hook ctl["haz"]/ctl["hu"] (a live agent dies iff hu < haz*eat). Brains drive all actions except eat for cohort agents.
Policies: COND (eat=1 iff own E < THETA), BLIND(e) constant eat, SHAM (COND rule on a random other live cohort agent's E, redrawn each step),
BRAIN (no override). Within-world pairing: COND on one uid parity of the cohort, comparator on the other, both parities.
  python exp25_run.py --procs 4 [--haz 0.004,0.0] [--seeds 0,...,8]     (resumable: one pickle per (haz, seed) in runs25/)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs25"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
T = 300; THETA = 0.40; TF = 1500
EGRID = [round(0.1 * i, 1) for i in range(11)]
COMPS = [("blind", e) for e in EGRID] + [("sham", 0.0), ("brain", 0.0)]

def build(seed):
    import jax, jax.numpy as jnp, exp19_run as X
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{TF}.npz"))
    base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}
    def run_one(s, kx, haz, grpA, mode, ce):
        """grpA: (N,) bool cohort agents on COND; other cohort agents on comparator (mode 0 blind ce, 1 sham, 2 brain)."""
        coh = s["alive"]; uid0 = s["uid"]; grpB = coh & ~grpA
        def body(c, k):
            k1, k2 = jax.random.split(k)
            same = c["alive"] & coh & (c["uid"] == uid0); E = c["E"]
            lc = same.astype(jnp.float32); idx = jax.random.choice(k1, N, (N,), p=lc / jnp.maximum(lc.sum(), 1.0))
            e_cond = (E < THETA).astype(jnp.float32); e_sham = (E[idx] < THETA).astype(jnp.float32)
            e_comp = jnp.where(mode == 0, ce, e_sham)
            use = same & (grpA | (mode != 2)); eov = jnp.where(grpA, e_cond, e_comp)
            hu = jax.random.uniform(k2, (N,))
            ctl = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0),
                       eat_use=use, eat_over=eov, haz=haz, hu=hu)
            c2 = step(c, prm, ctl)
            eat_eff = jnp.where(use, eov, (c2["act"][:, 2] + 1) / 2)
            pred = same & (hu < haz * eat_eff)
            now = c2["alive"] & coh & (c2["uid"] == uid0); died = same & ~now
            r = lambda m: jnp.stack([(m & grpA).sum(), (m & grpB).sum()])
            rf = lambda v: jnp.stack([jnp.where(same & grpA, v, 0.0).sum(), jnp.where(same & grpB, v, 0.0).sum()])
            return c2, dict(alive=r(now), pred=r(pred), starve=r(died & ~pred), eat=rf(eat_eff), Esum=jnp.stack([jnp.where(now & grpA, c2["E"], 0.0).sum(), jnp.where(now & grpB, c2["E"], 0.0).sum()]))
        _, rec = jax.lax.scan(body, s, jax.random.split(kx, T)); return rec
    return jax.jit(run_one), base

def run_job(job):
    haz, seed = job
    import jax, jax.numpy as jnp
    os.makedirs(OUT, exist_ok=True); path = os.path.join(OUT, f"run25_h{haz}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time(); f, base = build(seed); bj = {k: jnp.asarray(v) for k, v in base.items()}
    coh = base["alive"]; par = (base["uid"] % 2 == 0)
    kx = jax.random.PRNGKey(2500 + seed); res = {}
    for p in (0, 1):
        gA = jnp.asarray(coh & (par if p == 0 else ~par))
        for (m, e) in COMPS:
            mode = {"blind": 0, "sham": 1, "brain": 2}[m]
            rec = f(bj, kx, jnp.float32(haz), gA, jnp.int32(mode), jnp.float32(e))
            res[(p, m, e)] = {k: np.asarray(v) for k, v in rec.items()}
    n = dict(A0=[int((coh & par).sum()), int((coh & ~par).sum())])
    pickle.dump(dict(haz=haz, seed=seed, theta=THETA, T=T, comps=COMPS, nA={0: int((coh & par).sum()), 1: int((coh & ~par).sum())},
                     nB={0: int((coh & ~par).sum()), 1: int((coh & par).sum())}, res=res, secs=time.time() - t0), open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path); return path, time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--haz", default="0.004,0.0")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); a = ap.parse_args()
    jobs = [(float(h), int(s)) for h in a.haz.split(",") for s in a.seeds.split(",")]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for path, sec in pool.imap_unordered(run_job, jobs):
            print(time.strftime("%H:%M:%S"), os.path.basename(path), f"{sec:.0f}s", flush=True)
