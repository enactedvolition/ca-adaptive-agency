"""Experiment 6: score recovery on the variables an agent would regulate, not on hidden-state distance.

Reuses the full-state sim5 forks of Exp 5 (runs5/fork_*.npz, step 2500) and the same matched targets
(same selection RNG as exp5_perturb.py). For each run (variant x seed) and target, closed-loop forks of the
whole world (sim6 = sim5 + an observer-side intervention hook):
  ref        unperturbed (same PRNG key)
  twin/twin2 unperturbed, weather/birth PRNG key changed  -> noise floor: how far E, intake, PE drift from
             ref through world chaos alone, with no perturbation of the target
  hk0.1/0.3  hidden-state kick (random unit direction, eps 0.1/0.3), scored on |dE|, d(cum intake), d(PE)
  ed_cl      energy drop: target loses 50% of its energy at t=0, closed loop (it senses energy, ch. 8)
  ed_clamp   same drop, actions clamped to the ref agent's actions (passive: body repeats what the unperturbed
             twin did; any deficit closure is passive)
  ed_blind   same drop, closed loop but the energy *sensor* reads E + dropped amount (cannot see the deficit)
  clamp0     no drop, actions clamped to ref (validation: must equal ref exactly), first 8 targets
Active energy regulation = ed_cl closes the deficit more than ed_clamp (paired, CI over seeds excludes 0);
ed_blind tells whether that runs through sensing the deficit.

  python exp6_regulate.py --procs 6      # checkpoint = one file per run in runs6/
"""
from __future__ import annotations
import argparse, os, sys, time, pickle, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP6_OUT", os.path.join(HERE, "runs6"))
VARIANTS = ["sw_pecost_pe", "sw_full", "sw_pecost", "sw_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
SEEDS = [0, 1, 2]
T = int(os.environ.get("EXP6_T", 120))
N_TGT = int(os.environ.get("EXP6_NTGT", 40))
DROP = 0.5
N_VAL = 8
BF = 14
REC = ("h", "act", "pos", "E", "alive", "uid", "c_int", "pestep", "cstep")

def worker(job):
    variant, seed = job
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)
    path = os.path.join(OUT, f"reg_{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    import jax, jax.numpy as jnp, sim6, exp5_price as E5
    t0 = time.time()
    cfg = sim6.Cfg(N=2048, L=1.0, G=64)
    init, chunk_one, colour, consts = sim6.make(cfg)
    step = consts["step"]; H = cfg.H; N = cfg.N
    prm = E5.prm_of(variant)
    z = np.load(os.path.join(HERE, "runs5", f"fork_{variant}_s{seed}.npz"))
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    base = {k: (np.asarray(z[k]).astype(sd.dtype) if k in z.files else np.zeros(sd.shape, sd.dtype)) for k, sd in tmpl.items()}
    base["anc"] = base["uid"].copy()
    # ---- identical target selection to exp5_perturb.py
    rng = np.random.default_rng(1000 + seed)
    al = base["alive"]; ln = base["lam_n"]
    lam = np.where(ln >= 50, base["lam_sum"] / np.maximum(ln, 1), np.nan)
    cand = np.nonzero(al & np.isfinite(lam))[0]
    qs = np.quantile(lam[cand], [0.25, 0.5, 0.75]); bins = np.digitize(lam[cand], qs)
    tg = []
    for q in range(4):
        cq = rng.permutation(cand[bins == q]); per_root = {}
        for c in cq:
            r = int(base["root"][c])
            if per_root.get(r, 0) >= 6: continue
            per_root[r] = per_root.get(r, 0) + 1; tg.append(c)
            if sum(per_root.values()) >= N_TGT // 4: break
    tg = np.array(tg); K = len(tg)
    rng2 = np.random.default_rng(2000 + seed)
    key0 = jnp.asarray(base["key"])
    keys = {"ref": np.asarray(key0), "twin": np.asarray(jax.random.fold_in(key0, 777)), "twin2": np.asarray(jax.random.fold_in(key0, 778))}
    tg_j = jnp.asarray(tg, jnp.int32)
    # fork = (name, target index, hkick (H,), drop frac, clamp, blind)
    def run_one(s, key, slot, hk, drop, clamp, blind, aov):
        s = dict(s); s["key"] = key
        s["h"] = s["h"].at[slot].add(hk)
        dE = s["E"][slot] * drop
        s["E"] = s["E"].at[slot].add(-dE)
        uid0 = s["uid"][slot]; onehot = jnp.arange(N) == slot
        def body(c, a_t):
            same = c["uid"][slot] == uid0
            ctl = dict(aclamp=onehot & clamp & same, aover=jnp.zeros((N, 8)).at[slot].set(a_t),
                       e_off=jnp.where(onehot & blind & same, dE, 0.0))
            c = step(c, prm, ctl)
            return c, {k: c[k][tg_j] for k in REC}
        _, recs = jax.lax.scan(body, s, aov)
        return recs
    run_b = jax.jit(jax.vmap(run_one, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    def run_forks(forks, aov_of):
        R = []
        for i in range(0, len(forks), BF):
            fb = list(forks[i:i + BF]); nreal = len(fb)
            fb += [("pad", -1, None, 0.0, False, False)] * (BF - nreal)
            args = (jnp.asarray(np.stack([keys["twin"] if f[0] == "twin" else keys["twin2"] if f[0] == "twin2" else keys["ref"] for f in fb])),
                    jnp.asarray([tg[f[1]] if f[1] >= 0 else 0 for f in fb], jnp.int32),
                    jnp.asarray(np.stack([f[2] if f[2] is not None else np.zeros(H) for f in fb]), jnp.float32),
                    jnp.asarray([f[3] for f in fb], jnp.float32), jnp.asarray([f[4] for f in fb]), jnp.asarray([f[5] for f in fb]),
                    jnp.asarray(np.stack([aov_of(f) for f in fb]), jnp.float32))
            rec = run_b(base_j, *args)
            R.append({k: np.asarray(v)[:nreal] for k, v in rec.items()})
        return {k: np.concatenate([r[k] for r in R]) for k in R[0]}
    z8 = np.zeros((T, 8), np.float32)
    rec0 = run_forks([("ref", -1, None, 0.0, False, False), ("twin", -1, None, 0.0, False, False), ("twin2", -1, None, 0.0, False, False)], lambda f: z8)
    ref_act = rec0["act"][0]                                   # (T, K, 8)
    forks = []
    for ti in range(K):
        for e in (0.1, 0.3):
            d = rng2.normal(size=H); forks.append((f"hk{e}", ti, e * d / np.linalg.norm(d), 0.0, False, False))
        forks.append(("ed_cl", ti, None, DROP, False, False))
        forks.append(("ed_clamp", ti, None, DROP, True, False))
        forks.append(("ed_blind", ti, None, DROP, False, True))
        if ti < N_VAL: forks.append(("clamp0", ti, None, 0.0, True, False))
    rec1 = run_forks(forks, lambda f: ref_act[:, f[1]] if f[1] >= 0 else z8)
    ar = np.arange(K); uid0 = base["uid"][tg]
    def pick(rec, idx, ti):
        return {k: rec[k][idx][:, :, ti, ...][np.arange(len(idx)), :, np.arange(len(idx))] for k in rec}   # (n, T, ...)
    def glob(i):
        return {k: rec0[k][i].swapaxes(0, 1) for k in rec0}                                                # (K, T, ...)
    res = dict(variant=variant, seed=seed, targets=tg, lam=lam[tg], root=base["root"][tg], uid=uid0, E0=base["E"][tg], T=T, drop=DROP, conds={})
    for nm, i in (("ref", 0), ("twin", 1), ("twin2", 2)):
        res["conds"][nm] = glob(i)
    for nm in ("hk0.1", "hk0.3", "ed_cl", "ed_clamp", "ed_blind", "clamp0"):
        idx = [j for j, f in enumerate(forks) if f[0] == nm]; ti = np.array([forks[j][1] for j in idx])
        d = pick(rec1, idx, ti); d["ti"] = ti; res["conds"][nm] = d
    for nm in res["conds"]:
        c = res["conds"][nm]; c["own"] = c["alive"] & (c["uid"] == uid0[c.get("ti", ar)][:, None])
        for k in ("h", "act", "pos"):
            if nm not in ("ref",): c.pop(k, None) if k == "h" and nm.startswith("ed") else None
    v = res["conds"]["clamp0"]; r = res["conds"]["ref"]
    res["clamp0_maxdiff"] = float(np.abs(v["E"] - r["E"][v["ti"]]).max())
    res["wall_s"] = time.time() - t0; res["n_forks"] = len(forks) + 3
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, res["wall_s"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--only", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    if a.only:
        v, s = a.only.split(":"); print(worker((v, int(s)))); sys.exit()
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for p, w in pool.imap_unordered(worker, jobs):
            print("DONE", os.path.basename(p), f"{w:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(wall_s=time.time() - t0, procs=a.procs), open(os.path.join(OUT, "meta.json"), "w"))
