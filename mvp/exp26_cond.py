"""Experiment 26 part 1: PRE-REGISTERED REPLICATION of the Exp 25 assay positive control (survival read-out, K=200 targets, FRESH target rng 2626+seed and world/hazard key 2700+seed). Logic otherwise identical to exp25_assay.py. Original docstring:
Exp 25 ASSAY POSITIVE CONTROL. The Exp 22 births-off energy-drop assay run on a population whose eat
effort is the hand-coded COND rule (eat = 1 iff E < 0.40) under predation haz = 0.004 (sim13 ctl haz/hu), brains drive the other actions.
40 targets = cohort agents with E0 >= 0.3 (rng 2525 + seed), 50% E drop at step 0, T = 150, every agent's rep off.
Arms: ref, cl, open (all 8 brain actions AND eat replayed from ref while ref target alive), frozen (rule and brain sense ref's E trajectory),
clamp0 (no drop + open clamp; must equal ref). Hazard uniforms identical across arms. Hand-coded rule: a pass means the assay detects a
known energy-feedback controller, not agency of any evolved population.   python exp25_assay.py --procs 6   (resumable, runs26/cond26_s*.pkl)"""
import os, sys, time, pickle, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs26"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import exp25_run as R
T, DROP, E_MIN, HAZ = 150, 0.5, 0.3, 0.004
K_T = 200; SUF = ""   # EXP25_K: post hoc power check (more targets)

def run_seed(seed):
    os.makedirs(OUT, exist_ok=True)
    import jax, jax.numpy as jnp, exp19_run as X
    path = os.path.join(OUT, f"cond26{SUF}_s{seed}.pkl")
    if os.path.exists(path): return path, 0.0
    t0 = time.time()
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); step = consts["step"]; N = cfg.N
    z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t{R.TF}.npz"))
    base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}
    elig = np.nonzero(base["alive"] & (base["E"] >= E_MIN))[0]
    tg = np.sort(np.random.default_rng(2626 + seed).choice(elig, size=min(K_T, len(elig)), replace=False)); K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, kx, drop, clamp, frozen, aov, eov, eref):
        s = dict(s); coh = s["alive"]; uid0 = s["uid"]
        s["E"] = s["E"].at[tg_j].add(-drop * DROP * s["E"][tg_j])
        def body(c, x):
            k, a_t, ea_t, e_t = x
            same = c["alive"] & coh & (c["uid"] == uid0); sameT = same[tg_j]; E = c["E"]
            refal = jnp.abs(a_t).sum(1) > 0
            cl_T = clamp & sameT & refal; fr_T = frozen & sameT & (e_t >= 0)
            Erule = E.at[tg_j].set(jnp.where(fr_T, e_t, E[tg_j]))
            eov_all = (Erule < R.THETA).astype(jnp.float32).at[tg_j].set(jnp.where(cl_T, ea_t, (Erule[tg_j] < R.THETA).astype(jnp.float32)))
            hu = jax.random.uniform(k, (N,))
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(cl_T), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t),
                       e_off=jnp.zeros(N).at[tg_j].set(jnp.where(fr_T, e_t - E[tg_j], 0.0)), rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0),
                       eat_use=same, eat_over=eov_all, haz=jnp.float32(HAZ), hu=hu)
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0[tg_j])
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], eat=eov_all[tg_j], pred=sameT & (hu[tg_j] < HAZ * eov_all[tg_j]))
        _, rec = jax.lax.scan(body, s, (jax.random.split(kx, T), aov, eov, eref)); return rec
    f = jax.jit(run_one); bj = {k: jnp.asarray(v) for k, v in base.items()}; kx = jax.random.PRNGKey(2700 + seed)
    z8 = jnp.zeros((T, K, 8)); zk = jnp.zeros((T, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    g = lambda r: {k: np.asarray(v) for k, v in r.items()}
    ref = g(f(bj, kx, 0.0, F, F, z8, zk, zk - 1))
    a_ref = jnp.asarray(np.where(ref["alive"][..., None], ref["act"], 0.0)); ea_ref = jnp.asarray(ref["eat"]); e_ref = jnp.asarray(np.where(ref["alive"], ref["E"], -1.0))
    arms = dict(ref=ref, cl=g(f(bj, kx, 1.0, F, F, z8, zk, zk - 1)), open=g(f(bj, kx, 1.0, Tr, F, a_ref, ea_ref, zk - 1)),
                frozen=g(f(bj, kx, 1.0, F, Tr, z8, zk, e_ref)), clamp0=g(f(bj, kx, 0.0, Tr, F, a_ref, ea_ref, zk - 1)))
    pickle.dump(dict(seed=seed, tg=tg, E0=base["E"][tg], arms=arms, clamp0_maxdiff_E=float(np.abs(arms["clamp0"]["E"] - ref["E"]).max()), secs=time.time() - t0),
                open(path + ".tmp", "wb")); os.replace(path + ".tmp", path); return path, time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8"); a = ap.parse_args()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for path, sec in pool.imap_unordered(run_seed, [int(s) for s in a.seeds.split(",")]):
            print(time.strftime("%H:%M:%S"), os.path.basename(path), f"{sec:.0f}s", flush=True)
