"""Experiment 13: long horizon for the Exp 12 lead (NIGHTLY_LOG 2026-11-12 next step).
Same sim9 harness/params as exp12_run.py (claim_mode="mass", switching world sw_period=300, Exp 5 base, N=2048, G=64,
PRNGKey(1000*s+7), 10-step chunks, exact Price decomposition every 10 steps), but STEPS=10000 and seeds 0-8
(three vmapped batches of 3: EXP13_SEEDS="0,1,2" / "3,4,5" / "6,7,8"). Arms: sw_full, sw_mg2, sw_mg2_shuffled.
Pre-registered test (fixed 2026-11-12, before any 10k result existed): sw_mg2 - sw_mg2_shuffled on mean build over
t>=5000, paired by seed, n=9, 95% t CI > 0. Secondary: Price sum S(build) over t>=5000 for the same contrast; sw_mg2 - sw_full.
Checkpoints: full state + Price bookkeeping every CK steps to runs13/ck13_{v}_{tag}.pkl (resume is bit-for-bit, see
--resumecheck). Forks of the full state at t=5000 and t=10000 (runs13/fork13_*.npz, NOT committed).
  python exp13_run.py --resumecheck ; EXP13_SEEDS=0,1,2 python exp13_run.py --procs 3
"""
from __future__ import annotations
import argparse, os, sys, time, json, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exp12_run as E12
OUT = os.environ.get("EXP13_OUT", os.path.join(HERE, "runs13"))
STEPS = int(os.environ.get("EXP13_STEPS", "10000")); CK = int(os.environ.get("EXP13_CK", "1000")); DT = 10
FORKS = (5000, 10000)
SEEDS = [int(x) for x in os.environ.get("EXP13_SEEDS", "0,1,2").split(",")]
TAG = "s" + "".join(map(str, SEEDS))
VARIANTS = E12.VARIANTS; TRAITS = E12.TRAITS
KEYS_TS = ("t", "pop", "build", "claim", "intake", "E", "wall", "live_frac", "own_share_here", "pe", "prem", "bcost", "own_here")

def run(v, steps=STEPS, out=OUT, ck_every=CK, stop_at=None, forks=FORKS, prefix="run13"):
    E12.setup()
    import jax, jax.numpy as jnp, sim9
    path = os.path.join(out, f"{prefix}_{v}_{TAG}.pkl"); ckp = os.path.join(out, f"ck13_{v}_{TAG}.pkl")
    if os.path.exists(path): return v, 0.0, "cached"
    t0 = time.time(); B = len(SEEDS)
    cfg, st, bchunk, prm = E12.make(sim9, v, SEEDS)
    if os.path.exists(ckp):
        ck = pickle.load(open(ckp, "rb"))
        st = {k: jnp.asarray(x) for k, x in ck["st"].items()}; series, prev, t_start = ck["series"], ck["prev"], ck["t"]
        print(v, "resumed at", t_start, flush=True)
    else:
        series = [dict(**{k: [] for k in KEYS_TS}, n=[], n1=[], **{f"{q}_{c}": [] for q in TRAITS for c in ("zbar", "zbar1", "D", "S", "T", "W", "I", "resid")}) for _ in range(B)]
        prev = [None] * B; t_start = 0
        al = np.asarray(st["alive"]); st = dict(st); st["anc"] = jnp.asarray(np.where(al, np.asarray(st["uid"]), -1).astype(np.int32))
    g = lambda k: np.asarray(st[k])
    cs = cfg.L / cfg.G
    for t in range(t_start + DT, steps + 1, DT):
        st = bchunk(st, prm, DT)
        al, uid, l0n, anc, gpar = g("alive"), g("uid"), g("l0n"), g("anc").astype(np.int64), g("gpar")
        tr = {q: g(k) for q, k in TRAITS.items()}
        anc_new = anc.copy()
        if t % 50 == 0:
            Wl = g("Wl"); Ad = g("Adep"); pos = g("pos"); act = g("act"); cl = g("clstep"); inta = g("intake"); En = g("E"); pe = g("pestep"); pr = g("premstep")
        for b in range(B):
            mem = al[b] & (l0n[b] >= 1); mi = np.nonzero(mem)[0]
            cur = dict(uid=uid[b, mi].astype(np.int64), anc=anc[b, mi], **{"z" + q: tr[q][b, mi].astype(np.float64) / l0n[b, mi] for q in TRAITS})
            S_ = series[b]; P_ = prev[b]
            if P_ is not None:
                idx = {int(x): i for i, x in enumerate(P_["uid"])}
                n, n1 = len(P_["uid"]), len(cur["uid"])
                pi = np.array([idx.get(int(x), -1) for x in cur["anc"]], dtype=np.int64)
                is_s = (cur["anc"] == cur["uid"]) & (pi >= 0); is_o = (~is_s) & (pi >= 0); is_i = pi < 0
                w = np.zeros(n); np.add.at(w, pi[~is_i], 1.0)
                S_["n"].append(n); S_["n1"].append(n1)
                for q in TRAITS:
                    zp = P_["z" + q]; zc = cur["z" + q]; zbar = zp.mean(); zbar1 = zc.mean()
                    Sx = (np.sum(w * zp) - w.mean() * n * zbar) / n1
                    Tx = np.sum(zc[is_o] - zp[pi[is_o]]) / n1
                    Wx = np.sum(zc[is_s] - zp[pi[is_s]]) / n1
                    Ix = np.sum(zc[is_i] - zbar) / n1
                    D = zbar1 - zbar
                    for k, val in (("zbar", zbar), ("zbar1", zbar1), ("D", D), ("S", Sx), ("T", Tx), ("W", Wx), ("I", Ix), ("resid", D - (Sx + Tx + Wx + Ix))):
                        S_[f"{q}_{k}"].append(float(val))
            prev[b] = cur
            memset = set(cur["uid"].tolist())
            anc_new[b] = -1; anc_new[b, mi] = uid[b, mi]
            nm = np.nonzero(al[b] & ~mem)[0]
            anc_new[b, nm] = np.where(np.isin(gpar[b, nm], list(memset)), gpar[b, nm], -1)
            if t % 50 == 0:
                a = al[b]; ia = np.nonzero(a)[0]
                c = np.clip((pos[b, ia] / cs).astype(int), 0, cfg.G - 1)
                Wh = Wl[b, c[:, 0], c[:, 1]]; oh = Ad[b, ia, c[:, 0], c[:, 1]]
                for k, val in (("t", t), ("pop", int(a.sum())), ("build", float(((act[b, ia, 7] + 1) / 2).mean())), ("claim", float(cl[b, ia].mean())),
                               ("intake", float(inta[b, ia].mean())), ("E", float(En[b, ia].mean())), ("wall", float(Wl[b].sum())),
                               ("live_frac", float(Ad[b].sum() / max(Wl[b].sum(), 1e-12))), ("own_share_here", float(np.mean(oh / np.maximum(Wh, 1e-6)))),
                               ("pe", float(pe[b, ia].mean())), ("prem", float(pr[b, ia].mean())),
                               ("bcost", float((cfg.c_build * (act[b, ia, 7] + 1) / 2).mean())), ("own_here", float(np.mean(oh)))):
                    S_[k].append(val)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t in forks:
            for b, s in enumerate(SEEDS):
                d = {k: np.asarray(x[b]) for k, x in st.items()}
                tmp = os.path.join(out, f"fork13_{v}_s{s}_t{t}.tmp.npz"); np.savez_compressed(tmp, **d); os.replace(tmp, os.path.join(out, f"fork13_{v}_s{s}_t{t}.npz"))
        if t % ck_every == 0 and t < steps:
            ck = dict(t=t, st={k: np.asarray(x) for k, x in st.items()}, series=series, prev=prev)
            pickle.dump(ck, open(ckp + ".tmp", "wb"), protocol=5); os.replace(ckp + ".tmp", ckp)
            print(v, TAG, t, f"{time.time()-t0:.0f}s ckpt", flush=True)
            if stop_at is not None and t >= stop_at: return v, time.time() - t0, "stopped"
    res = dict(variant=v, params=dict(E12.BASE, **VARIANTS[v]), seeds=SEEDS, DT=DT, steps=steps,
               series=[{k: np.asarray(x) for k, x in s.items()} for s in series], wall_s=time.time() - t0)
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    if os.path.exists(ckp): os.remove(ckp)
    return v, res["wall_s"], "ok"

def resumecheck(steps=200):
    """straight run to `steps` vs run that checkpoints at steps/2, stops, and resumes in a fresh call: series + state must match bit for bit."""
    import shutil
    d1 = os.path.join(OUT, "rc_straight"); d2 = os.path.join(OUT, "rc_resume")
    for d in (d1, d2): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    run("sw_mg2", steps, d1, ck_every=10**9, forks=(steps,), prefix="rc")
    run("sw_mg2", steps, d2, ck_every=steps // 2, stop_at=steps // 2, forks=(steps,), prefix="rc")
    run("sw_mg2", steps, d2, ck_every=steps // 2, forks=(steps,), prefix="rc")
    a = pickle.load(open(os.path.join(d1, f"rc_sw_mg2_{TAG}.pkl"), "rb"))["series"]; b = pickle.load(open(os.path.join(d2, f"rc_sw_mg2_{TAG}.pkl"), "rb"))["series"]
    ser_ok = all(np.array_equal(a[i][k], b[i][k]) for i in range(len(a)) for k in a[i])
    st_ok = True
    for s in SEEDS:
        x = np.load(os.path.join(d1, f"fork13_sw_mg2_s{s}_t{steps}.npz")); y = np.load(os.path.join(d2, f"fork13_sw_mg2_s{s}_t{steps}.npz"))
        st_ok &= all(np.array_equal(x[k], y[k], equal_nan=True) for k in x.files)
    r = dict(steps=steps, series_bit_for_bit=bool(ser_ok), state_bit_for_bit=bool(st_ok), n_series_len=len(a[0]["a_S"]))
    json.dump(r, open(os.path.join(OUT, "resumecheck.json"), "w"), indent=1); print(json.dumps(r), flush=True)
    for d in (d1, d2): shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=3); ap.add_argument("--resumecheck", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.resumecheck: resumecheck(); sys.exit(0)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for v, w, d in pool.imap_unordered(run, list(VARIANTS)):
            print("DONE", v, TAG, f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
