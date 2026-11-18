"""Experiment 16, post hoc / exploratory (specified in NIGHTLY_LOG 2026-11-18 01:33 PST before running): mixed-competition version of the
pay-check. rep_free = 1 world (sim12), started from the Exp 15b sw_full t=3000 forks (seeds 0-2), 400 steps; brains drive every action
except rep. Policy by lineage: root even -> aware (rep = +1 iff true E > thr), root odd -> always (rep = +1; the simulator's 0.35 gate
still applies). Offspring inherit root from the genome donor, so policy is heritable; background spawns get fresh roots.
Outcome: aware lineages' population share at t=0 vs mean over the last 50 steps.   python exp16_compete.py"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs16")
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; sys.path.insert(0, HERE)
import jax, jax.numpy as jnp, sim12, exp16_run as E16
THRS = (0.8, 1.2, 1.6); STEPS = 400; SEEDS = (0, 1, 2)
cfg = sim12.Cfg(N=2048, L=1.0, G=64, **E16.CFGKW); init, chunk_one, colour, consts = sim12.make(cfg); step = consts["step"]; N = cfg.N
prm = sim12.params(**dict(E16.BASE, sw_period=E16.SW, rep_free=1.0))
tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
def state0(seed):
    z = np.load(os.path.join(E16.FORK15, f"fork15_sw_full_s{seed}_t3000.npz"))
    return {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) if k in z.files else jnp.zeros(sd.shape, sd.dtype) for k, sd in tmpl.items()}
z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
@jax.jit
def simj(s, thr):
    def body(c, _):
        aw = (c["root"] % 2) == 0
        ro = jnp.where(aw, jnp.where(c["E"] > thr, 1.0, -1.0), 1.0)
        c2 = step(c, prm, dict(aclamp=zb, aover=z8, e_off=zf, rep_use=jnp.ones(N, bool), rep_over=ro))
        al = c2["alive"]; a2 = al & ((c2["root"] % 2) == 0)
        return c2, dict(pop=al.sum(), aw=a2.sum(), b_aw=(c2["repstep"] * aw).sum(), b_al=(c2["repstep"] * ~aw).sum(),
                        E_aw=(c2["E"] * a2).sum() / jnp.maximum(a2.sum(), 1), E_al=(c2["E"] * (al & ~a2)).sum() / jnp.maximum((al & ~a2).sum(), 1))
    _, r = jax.lax.scan(body, s, None, length=STEPS)
    return r
out = dict(steps=STEPS, seeds=list(SEEDS), thrs=list(THRS), per={})
for thr in THRS:
    rows = []
    for sd in SEEDS:
        s = state0(sd); al = np.asarray(s["alive"]); sh0 = float(((np.asarray(s["root"]) % 2 == 0) & al).sum() / al.sum())
        r = {k: np.asarray(v) for k, v in simj(s, jnp.float32(thr)).items()}
        shT = float((r["aw"][-50:] / r["pop"][-50:]).mean())
        row = dict(seed=sd, share0=sh0, shareT=shT, dshare=shT - sh0, popT=float(r["pop"][-50:].mean()), births_aware=float(r["b_aw"].sum()),
                   births_always=float(r["b_al"].sum()), E_aware=float(r["E_aw"][-50:].mean()), E_always=float(r["E_al"][-50:].mean()),
                   share_curve=(r["aw"] / np.maximum(r["pop"], 1))[::10].round(4).tolist())
        rows.append(row); print(thr, json.dumps({k: v for k, v in row.items() if k != "share_curve"}), flush=True)
    out["per"][str(thr)] = rows
    out[f"dshare_{thr}"] = [r["dshare"] for r in rows]
json.dump(out, open(os.path.join(OUT, "compete.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.startswith("dshare")}), flush=True)
