"""Experiment 4b: perturbation recovery from full-state sim4 forks (exp4_price.py, step FORK_T), same protocol as exp3b.

For each run (variant x seed) we pick matched living agents (stratified by lambda quartile, <=6 per lineage per quartile),
fork the *whole* simulation state, perturb only the target's hidden state, and resume closed-loop dynamics
(its actions change its position, harvest, neighbours, fields -> its inputs). Conditions per target:
  kick_eps (random unit direction, eps in EPS), kick_lyap (eps=0.1 along the saved tangent = top Lyapunov
  vector estimate), ongoing (i.i.d. N(0, SIG^2) added to h every step).
One unperturbed fork per run is the reference (same PRNG key -> same weather / births unless the target's
behaviour changes them). Open-loop (recorded unperturbed inputs) and frozen-input (input held at the fork value)
replays are computed host-side for the same brain + perturbation.

  python exp4_perturb.py --procs 8      # checkpoint = one file per run in runs4/perturb/
"""
from __future__ import annotations
import argparse, os, sys, time, pickle, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs4", "perturb")
VARIANTS = ["full", "shuffled", "no_cost", "static_env", "pe", "lamcost", "pe_lamcost"]
SEEDS = [0, 1, 2]
T = 100
EPS = [0.03, 0.1, 0.3]
EPS_LYAP = 0.1
SIG = 0.01
N_TGT = 40
BF = 14                         # forks per vmapped call

def worker(job):
    variant, seed = job
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)
    path = os.path.join(OUT, f"pert_{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    import jax, jax.numpy as jnp, sim4 as sim3, exp4_price as E4
    t0 = time.time()
    cfg = sim3.Cfg(N=2048, L=1.0, G=64)
    init, chunk_one, colour, consts = sim3.make(cfg)
    step = consts["step"]; lay = consts["lay"]; H = cfg.H; N = cfg.N
    prm = E4.prm_of(variant)
    z = np.load(os.path.join(HERE, "runs4", f"fork_{variant}_s{seed}.npz"))
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    base = {}
    for k, sd in tmpl.items():
        if k in z.files:
            base[k] = np.asarray(z[k]).astype(sd.dtype)
        else:
            base[k] = np.zeros(sd.shape, sd.dtype)
    base["anc"] = base["uid"].copy()
    # ---- target selection: alive, lam_n>=50, stratified by lambda quartile, <=6 per root per quartile
    rng = np.random.default_rng(1000 + seed)
    al = base["alive"]; ln = base["lam_n"]
    lam = np.where(ln >= 50, base["lam_sum"] / np.maximum(ln, 1), np.nan)
    cand = np.nonzero(al & np.isfinite(lam))[0]
    qs = np.quantile(lam[cand], [0.25, 0.5, 0.75])
    bins = np.digitize(lam[cand], qs)
    tg = []
    for q in range(4):
        cq = rng.permutation(cand[bins == q]); per_root = {}
        for c in cq:
            r = int(base["root"][c])
            if per_root.get(r, 0) >= 6:
                continue
            per_root[r] = per_root.get(r, 0) + 1; tg.append(c)
            if sum(per_root.values()) >= N_TGT // 4:
                break
    tg = np.array(tg); K = len(tg)
    # ---- fork list
    forks = [("ref", -1, None)]
    for ti, sl in enumerate(tg):
        for e in EPS:
            d = rng.normal(size=H); forks.append((f"kick{e}", ti, e * d / np.linalg.norm(d)))
        v = base["v_tan"][sl]; forks.append(("lyap", ti, EPS_LYAP * v / np.linalg.norm(v)))
        forks.append(("ongoing", ti, rng.normal(size=(T, H)) * SIG))
    def noise_of(f):
        nz = np.zeros((T, H), np.float32)
        if f[0] == "ongoing":
            nz[:] = f[2]
        elif f[0] != "ref":
            nz[0] = f[2]
        return nz
    tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, slot, noise):
        def body(c, nz):
            c = dict(c); c["h"] = c["h"].at[slot].add(nz)
            c = step(c, prm)
            rec = {k: c[k][tg_j] for k in ("h", "act", "pos", "E", "alive", "uid", "xprev", "lstep", "cstep")}
            return c, rec
        _, recs = jax.lax.scan(body, s, noise)
        return recs
    run_b = jax.jit(jax.vmap(run_one, in_axes=(None, 0, 0)))
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    R = []
    for i in range(0, len(forks), BF):
        fb = forks[i:i + BF]
        slots = jnp.asarray([tg[f[1]] if f[1] >= 0 else 0 for f in fb], jnp.int32)
        noise = jnp.asarray(np.stack([noise_of(f) for f in fb]))
        if len(fb) < BF:                        # pad to a fixed batch (one compile)
            pad = BF - len(fb)
            slots = jnp.concatenate([slots, jnp.zeros(pad, jnp.int32)]); noise = jnp.concatenate([noise, jnp.zeros((pad, T, H))])
        rec = run_b(base_j, slots, noise)
        rec = {k: np.asarray(v)[:len(fb)] for k, v in rec.items()}
        R.append(rec)
    rec = {k: np.concatenate([r[k] for r in R]) for k in R[0]}           # (F, T, K, ...)
    # ---- host-side open-loop / frozen-input replays of the same brains
    W = base["W"][tg]
    g = {n: W[:, o:o + zz].reshape((K,) + sh) for n, sh, o, zz in lay}
    x_ref = rec["xprev"][0]                    # (T, K, 16) inputs of the unperturbed run
    x0 = base["xprev"][tg]                     # input at the fork
    h0 = base["h"][tg].astype(np.float64)
    def brain(hs, xs, kick=None, noise=None, v=None):
        """hs (K,H) start; xs (T,K,15) or (K,15) frozen. returns h (T,K,H), local log-stretch (T,K)."""
        h = hs.copy(); out = []; ls = []; vv = v.copy() if v is not None else None
        for t in range(T):
            if noise is not None:
                h = h + noise[t]
            x = xs[t] if xs.ndim == 3 else xs
            hn = np.tanh(np.einsum("ki,kih->kh", x, g["W_ih"]) + g["b_h"] + np.einsum("kj,kjh->kh", h, g["W_hh"]))
            if vv is not None:
                jv = (1 - hn**2) * np.einsum("kj,kjh->kh", vv, g["W_hh"]); n = np.linalg.norm(jv, axis=1); vv = jv / n[:, None]; ls.append(np.log(n))
            h = hn; out.append(h)
        return np.stack(out), (np.stack(ls) if ls else None)
    v0 = base["v_tan"][tg].astype(np.float64)
    ol_ref, ol_ref_ls = brain(h0, x_ref, v=v0)
    fr_ref, _ = brain(h0, x0)
    al_ref = rec["alive"][0] & (rec["uid"][0] == base["uid"][tg][None])
    match_err = float(np.abs(ol_ref - rec["h"][0])[:10][al_ref[:10]].max())   # host recursion vs sim, first 10 steps, while alive
    res = dict(variant=variant, seed=seed, targets=tg, lam=lam[tg], root=base["root"][tg], uid=base["uid"][tg], E0=base["E"][tg],
               match_err=match_err, conds={})
    ar = np.arange(K)
    for name in ["kick0.03", "kick0.1", "kick0.3", "lyap", "ongoing"]:
        idx = [i for i, f in enumerate(forks) if f[0] == name]
        ti = np.array([forks[i][1] for i in idx])          # target index per fork (== 0..K-1)
        nzs = np.stack([noise_of(forks[i]) for i in idx]).transpose(1, 0, 2)   # (T, K, H)
        cl = {k: rec[k][idx][:, :, ti, ...][ar, :, ar] for k in rec}           # pick own target -> (K, T, ...)
        ref = {k: rec[k][0][:, ti, ...].swapaxes(0, 1) for k in rec}          # (K, T, ...)
        kick = nzs[0] if name != "ongoing" else None
        if name == "ongoing":
            ol, ol_ls = brain(h0, x_ref, noise=nzs, v=v0); fr, _ = brain(h0, x0, noise=nzs)
        else:
            ol, ol_ls = brain(h0 + kick, x_ref, v=v0); fr, _ = brain(h0 + kick, x0)
        same = cl["uid"] == ref["uid"][:, :1]
        alive_cl = cl["alive"] & (cl["uid"] == ref["uid"][:, :1])
        alive_ref = ref["alive"] & (ref["uid"] == ref["uid"][:, :1])
        dpos = cl["pos"] - ref["pos"]; dpos -= np.round(dpos)          # torus L=1
        res["conds"][name] = dict(
            d_cl=np.linalg.norm(cl["h"] - ref["h"], axis=-1),                         # (K, T)
            d_ol=np.linalg.norm(ol - ol_ref, axis=-1).T,
            d_fr=np.linalg.norm(fr - fr_ref, axis=-1).T,
            d0=np.linalg.norm(kick, axis=-1) if kick is not None else np.full(K, np.nan),
            lam_cl=cl["lstep"], lam_ref=ref["lstep"], lam_ol=ol_ls.T, lam_olref=ol_ref_ls.T,
            d_act=np.linalg.norm(cl["act"] - ref["act"], axis=-1), d_pos=np.linalg.norm(dpos, axis=-1),
            E_cl=cl["E"], E_ref=ref["E"], alive_cl=alive_cl, alive_ref=alive_ref,
            cost_cl=cl["cstep"] * alive_cl, cost_ref=ref["cstep"] * alive_ref)
    res["wall_s"] = time.time() - t0; res["n_forks"] = len(forks)
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
