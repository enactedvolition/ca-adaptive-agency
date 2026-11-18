"""Experiment 16 perturbation protocols (NIGHTLY_LOG 2026-11-18): Exp 6 energy-drop test (phase B: closed loop vs actions clamped
to the unperturbed twin vs sensor-blind; PRIMARY = B = r_cl - r_blind in sw_pays at t=3000, n=9) and the Exp 5 kick protocol (phase A:
closed loop vs open-loop vs frozen-input replay) on the Exp 16 sim12 forks (runs16/fork16_{v}_s{s}_t{T}.npz), arms sw_full, sw_pays,
sw_pays_blind, seeds 0-8, T = 3000 (primary) and 2500. Code identical to exp14_perturb.py except the simulator (sim12, with each arm's
own params) and the fork source. Same target selection (rng 1000+seed, lambda quartiles, <=6 per root per quartile, 40 targets).
  python exp16_perturb.py --procs 2 --phases B,A --horizons 3000,2500
"""
from __future__ import annotations
import argparse, os, sys, time, pickle, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP16P_OUT", os.path.join(HERE, "runs16p"))
FORKS = os.environ.get("EXP16_FORKS", os.path.join(HERE, "runs16"))
VARIANTS = ["sw_pays", "sw_full", "sw_pays_blind"]
SEEDS = list(range(9))
HORIZONS = [3000, 2500]
T_A = 100; T_B = 120
EPS = [0.03, 0.1, 0.3]; EPS_LYAP = 0.1; SIG = 0.01
N_TGT = 40; DROP = 0.5; N_VAL = 8
BF = int(os.environ.get("EXP14_BF", 6))
REC_A = ("h", "act", "pos", "E", "alive", "uid", "xprev", "lstep", "cstep", "pestep")
REC_B = ("h", "act", "pos", "E", "alive", "uid", "c_int", "pestep", "cstep")

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def load(variant, seed, tf):
    import jax, sim12, exp16_run as E16
    cfg = sim12.Cfg(N=2048, L=1.0, G=64, **E16.CFGKW)
    init, chunk_one, colour, consts = sim12.make(cfg)
    prm = sim12.params(**dict(E16.BASE, **E16.VARIANTS[variant]))
    fp = os.path.join(FORKS, f"fork16_{variant}_s{seed}_t{tf}.npz")
    z = np.load(fp)
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    missing = [k for k in tmpl if k not in z.files]; extra = [k for k in z.files if k not in tmpl]
    base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items() if k in z.files}
    shape_ok = all(base[k].shape == tuple(sd.shape) for k, sd in tmpl.items() if k in base)
    exact = all(np.array_equal(base[k], z[k], equal_nan=True) and base[k].dtype == z[k].dtype for k in base)
    chk = dict(missing=missing, extra=extra, shape_ok=bool(shape_ok), exact=bool(exact), t=int(base["t"]), pop=int(base["alive"].sum()))
    return cfg, consts, prm, base, chk

def targets(base, seed):
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
    return np.array(tg), lam, rng

def phase_a(variant, seed, tf):
    """Exp 5 protocol (exp5_perturb.py), unchanged except sim9 forks."""
    import jax, jax.numpy as jnp
    t0 = time.time()
    cfg, consts, prm, base, chk = load(variant, seed, tf)
    step = consts["step"]; lay = consts["lay"]; H = cfg.H; T = T_A
    tg, lam, rng = targets(base, seed); K = len(tg)
    forks = [("ref", -1, None)]
    for ti, sl in enumerate(tg):
        for e in EPS:
            d = rng.normal(size=H); forks.append((f"kick{e}", ti, e * d / np.linalg.norm(d)))
        v = base["v_tan"][sl]; forks.append(("lyap", ti, EPS_LYAP * v / np.linalg.norm(v)))
        forks.append(("ongoing", ti, rng.normal(size=(T, H)) * SIG))
    def noise_of(f):
        nz = np.zeros((T, H), np.float32)
        if f[0] == "ongoing": nz[:] = f[2]
        elif f[0] != "ref": nz[0] = f[2]
        return nz
    tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, slot, noise):
        def body(c, nz):
            c = dict(c); c["h"] = c["h"].at[slot].add(nz)
            c = step(c, prm)
            return c, {k: c[k][tg_j] for k in REC_A}
        _, recs = jax.lax.scan(body, s, noise)
        return recs
    run_b = jax.jit(jax.vmap(run_one, in_axes=(None, 0, 0)))
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    R = []
    for i in range(0, len(forks), BF):
        fb = forks[i:i + BF]
        slots = jnp.asarray([tg[f[1]] if f[1] >= 0 else 0 for f in fb], jnp.int32)
        noise = jnp.asarray(np.stack([noise_of(f) for f in fb]))
        if len(fb) < BF:
            pad = BF - len(fb)
            slots = jnp.concatenate([slots, jnp.zeros(pad, jnp.int32)]); noise = jnp.concatenate([noise, jnp.zeros((pad, T, H))])
        rec = run_b(base_j, slots, noise)
        R.append({k: np.asarray(v)[:len(fb)] for k, v in rec.items()})
    rec = {k: np.concatenate([r[k] for r in R]) for k in R[0]}
    # repeat-determinism check: the unperturbed ref recomputed in a second, differently-composed batch
    rep_rec = run_b(base_j, jnp.zeros(BF, jnp.int32), jnp.zeros((BF, T, H)))
    ref_repeat_equal = all(np.array_equal(np.asarray(rep_rec[k])[BF - 1], rec[k][0], equal_nan=True) for k in REC_A)
    W = base["W"][tg]
    g = {n: W[:, o:o + zz].reshape((K,) + sh) for n, sh, o, zz in lay}
    x_ref = rec["xprev"][0]; x0 = base["xprev"][tg]; h0 = base["h"][tg].astype(np.float64)
    def brain(hs, xs, noise=None, v=None):
        h = hs.copy(); out = []; ls = []; vv = v.copy() if v is not None else None
        for t in range(T):
            if noise is not None: h = h + noise[t]
            x = xs[t] if xs.ndim == 3 else xs
            hn = np.tanh(np.einsum("ki,kih->kh", x, g["W_ih"]) + g["b_h"] + np.einsum("kj,kjh->kh", h, g["W_hh"]))
            if vv is not None:
                jv = (1 - hn**2) * np.einsum("kj,kjh->kh", vv, g["W_hh"]); n = np.linalg.norm(jv, axis=1); vv = jv / n[:, None]; ls.append(np.log(n))
            h = hn; out.append(h)
        return np.stack(out), (np.stack(ls) if ls else None)
    v0 = base["v_tan"][tg].astype(np.float64)
    ol_ref, ol_ref_ls = brain(h0, x_ref, v=v0); fr_ref, _ = brain(h0, x0)
    al_ref = rec["alive"][0] & (rec["uid"][0] == base["uid"][tg][None])
    match_err = float(np.abs(ol_ref - rec["h"][0])[:10][al_ref[:10]].max())
    res = dict(variant=variant, seed=seed, tf=tf, phase="A", targets=tg, lam=lam[tg], root=base["root"][tg], uid=base["uid"][tg],
               E0=base["E"][tg], match_err=match_err, fork_check=chk, ref_repeat_equal=bool(ref_repeat_equal), conds={},
               ref_E=rec["E"][0], ref_h=rec["h"][0], ref_pos=rec["pos"][0], ref_alive=rec["alive"][0], ref_uid=rec["uid"][0])
    ar = np.arange(K)
    for name in ["kick0.03", "kick0.1", "kick0.3", "lyap", "ongoing"]:
        idx = [i for i, f in enumerate(forks) if f[0] == name]
        ti = np.array([forks[i][1] for i in idx])
        nzs = np.stack([noise_of(forks[i]) for i in idx]).transpose(1, 0, 2)
        cl = {k: rec[k][idx][:, :, ti, ...][ar, :, ar] for k in rec}
        ref = {k: rec[k][0][:, ti, ...].swapaxes(0, 1) for k in rec}
        kick = nzs[0] if name != "ongoing" else None
        if name == "ongoing":
            ol, ol_ls = brain(h0, x_ref, noise=nzs, v=v0); fr, _ = brain(h0, x0, noise=nzs)
        else:
            ol, ol_ls = brain(h0 + kick, x_ref, v=v0); fr, _ = brain(h0 + kick, x0)
        alive_cl = cl["alive"] & (cl["uid"] == ref["uid"][:, :1])
        alive_ref = ref["alive"] & (ref["uid"] == ref["uid"][:, :1])
        dpos = cl["pos"] - ref["pos"]; dpos -= np.round(dpos)
        res["conds"][name] = dict(
            d_cl=np.linalg.norm(cl["h"] - ref["h"], axis=-1), d_ol=np.linalg.norm(ol - ol_ref, axis=-1).T,
            d_fr=np.linalg.norm(fr - fr_ref, axis=-1).T, d0=np.linalg.norm(kick, axis=-1) if kick is not None else np.full(K, np.nan),
            lam_cl=cl["lstep"], lam_ref=ref["lstep"], lam_ol=ol_ls.T, lam_olref=ol_ref_ls.T,
            d_act=np.linalg.norm(cl["act"] - ref["act"], axis=-1), d_pos=np.linalg.norm(dpos, axis=-1),
            E_cl=cl["E"], E_ref=ref["E"], alive_cl=alive_cl, alive_ref=alive_ref,
            cost_cl=cl["cstep"] * alive_cl, cost_ref=ref["cstep"] * alive_ref, pe_cl=cl["pestep"] * alive_cl, pe_ref=ref["pestep"] * alive_ref)
    res["wall_s"] = time.time() - t0; res["n_forks"] = len(forks)
    return res

def phase_b(variant, seed, tf):
    """Exp 6 energy-drop test (exp6_regulate.py): ref, twin/twin2 (new weather key), hk0.1, ed_cl, ed_clamp, ed_blind, clamp0."""
    import jax, jax.numpy as jnp
    t0 = time.time()
    cfg, consts, prm, base, chk = load(variant, seed, tf)
    step = consts["step"]; H = cfg.H; N = cfg.N; T = T_B
    tg, lam, _ = targets(base, seed); K = len(tg)
    key0 = jnp.asarray(base["key"])
    keys = {"ref": np.asarray(key0), "twin": np.asarray(jax.random.fold_in(key0, 777)), "twin2": np.asarray(jax.random.fold_in(key0, 778))}
    tg_j = jnp.asarray(tg, jnp.int32)
    def run_one(s, key, slot, hk, drop, clamp, blind, aov):
        s = dict(s); s["key"] = key
        s["h"] = s["h"].at[slot].add(hk)
        dE = s["E"][slot] * drop
        s["E"] = s["E"].at[slot].add(-dE)
        uid0 = s["uid"][slot]; onehot = jnp.arange(N) == slot
        def body(c, a_t):
            same = c["uid"][slot] == uid0
            ctl = dict(aclamp=onehot & clamp & same, aover=jnp.zeros((N, 8)).at[slot].set(a_t), e_off=jnp.where(onehot & blind & same, dE, 0.0))
            c = step(c, prm, ctl)
            return c, {k: c[k][tg_j] for k in REC_B}
        _, recs = jax.lax.scan(body, s, aov)
        return recs
    run_b = jax.jit(jax.vmap(run_one, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    def run_forks(forks, aov_of):
        R = []
        for i in range(0, len(forks), BF):
            fb = list(forks[i:i + BF]); nreal = len(fb)
            fb += [("pad", -1, None, 0.0, False, False)] * (BF - nreal)
            args = (jnp.asarray(np.stack([keys.get(f[0], keys["ref"]) for f in fb])),
                    jnp.asarray([tg[f[1]] if f[1] >= 0 else 0 for f in fb], jnp.int32),
                    jnp.asarray(np.stack([f[2] if f[2] is not None else np.zeros(H) for f in fb]), jnp.float32),
                    jnp.asarray([f[3] for f in fb], jnp.float32), jnp.asarray([f[4] for f in fb]), jnp.asarray([f[5] for f in fb]),
                    jnp.asarray(np.stack([aov_of(f) for f in fb]), jnp.float32))
            rec = run_b(base_j, *args)
            R.append({k: np.asarray(v)[:nreal] for k, v in rec.items()})
        return {k: np.concatenate([r[k] for r in R]) for k in R[0]}
    z8 = np.zeros((T, 8), np.float32)
    rec0 = run_forks([("ref", -1, None, 0.0, False, False), ("twin", -1, None, 0.0, False, False), ("twin2", -1, None, 0.0, False, False)], lambda f: z8)
    ref_act = rec0["act"][0]
    rng2 = np.random.default_rng(2000 + seed)
    forks = []
    for ti in range(K):
        d = rng2.normal(size=H); forks.append(("hk0.1", ti, 0.1 * d / np.linalg.norm(d), 0.0, False, False))
        forks.append(("ed_cl", ti, None, DROP, False, False))
        forks.append(("ed_clamp", ti, None, DROP, True, False))
        forks.append(("ed_blind", ti, None, DROP, False, True))
        if ti < N_VAL: forks.append(("clamp0", ti, None, 0.0, True, False))
    rec1 = run_forks(forks, lambda f: ref_act[:, f[1]] if f[1] >= 0 else z8)
    ar = np.arange(K); uid0 = base["uid"][tg]
    def pick(rec, idx, ti):
        return {k: rec[k][idx][:, :, ti, ...][np.arange(len(idx)), :, np.arange(len(idx))] for k in rec}
    res = dict(variant=variant, seed=seed, tf=tf, phase="B", targets=tg, lam=lam[tg], root=base["root"][tg], uid=uid0, E0=base["E"][tg],
               T=T, drop=DROP, fork_check=chk, conds={})
    for nm, i in (("ref", 0), ("twin", 1), ("twin2", 2)):
        res["conds"][nm] = {k: rec0[k][i].swapaxes(0, 1) for k in rec0}
    for nm in ("hk0.1", "ed_cl", "ed_clamp", "ed_blind", "clamp0"):
        idx = [j for j, f in enumerate(forks) if f[0] == nm]; ti = np.array([forks[j][1] for j in idx])
        d = pick(rec1, idx, ti); d["ti"] = ti; res["conds"][nm] = d
    for nm in res["conds"]:
        c = res["conds"][nm]; c["own"] = c["alive"] & (c["uid"] == uid0[c.get("ti", ar)][:, None])
        if nm != "ref": c.pop("h", None); c.pop("act", None)
    v = res["conds"]["clamp0"]; r = res["conds"]["ref"]
    res["clamp0_maxdiff"] = float(np.abs(v["E"] - r["E"][v["ti"]]).max())
    res["wall_s"] = time.time() - t0; res["n_forks"] = len(forks) + 3
    return res

def worker(job):
    variant, seed, tf, ph = job
    setup()
    path = os.path.join(OUT, f"p16{ph}_{variant}_s{seed}_t{tf}.pkl")
    if os.path.exists(path): return path, 0.0, "cached"
    res = phase_a(variant, seed, tf) if ph == "A" else phase_b(variant, seed, tf)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(res, open(path + ".tmp", "wb"), protocol=5); os.replace(path + ".tmp", path)
    return path, res["wall_s"], "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); ap.add_argument("--only", default=None)
    ap.add_argument("--phases", default="B,A"); ap.add_argument("--horizons", default="3000,2500")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.only:
        v, s, tf, ph = a.only.split(":"); print(worker((v, int(s), int(tf), ph)), flush=True); sys.exit()
    # priority: primary (sw_full, t10000, phase A) first
    jobs = [(v, s, tf, ph) for ph in a.phases.split(",") for tf in map(int, a.horizons.split(",")) for v in VARIANTS for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for p, w, d in pool.imap_unordered(worker, jobs):
            print("DONE", os.path.basename(p), f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(wall_s=time.time() - t0, procs=a.procs), open(os.path.join(OUT, "meta.json"), "w"))
