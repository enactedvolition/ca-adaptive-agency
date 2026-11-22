"""Experiment 19: is energy-conditioned reproduction rewarded once it exists in mutable brains? Pre-registered in NIGHTLY_LOG 2026-11-21.
Start: Exp 17 sw_pays t=3000 forks (sim13, rep_free=1, rep_fixed=1). A random 50% of live agents get one hidden unit rewired to the
aware rule (rep > 0 iff sensed E > 0.8). Arms: TRUE (implanted agents sense own E) vs SHAM (implanted agents' energy channel reads a
random other live agent's E, via sim13's ctl["e_off"]). 1500 steps, mutation on, exact Price (residual logged) on m (implant marker)
and a = c_esr/l0n. No sim change.
  python exp19_run.py --check                      ctl path with e_off=0 == plain sim13 step; implant rule exact at t=0
  EXP19_SEEDS=0,1,2 python exp19_run.py true       (one arm per process; rerun to resume from checkpoint; finished runs skipped)"""
import os, sys, time, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("EXP19_OUT", os.path.join(HERE, "runs19")); sys.path.insert(0, HERE)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
STEPS = int(os.environ.get("EXP19_STEPS", "1500")); CK = 500; DT = 10
SEEDS = [int(x) for x in os.environ.get("EXP19_SEEDS", "0,1,2").split(",")]; TAG = "s" + "".join(map(str, SEEDS))
FORK = os.path.join(HERE, "runs17"); G_IN, W_OUT, THR = 10.0, 3.0, 0.8; MARK = 0.5 * G_IN * W_OUT
BASE = dict(c_basal=0.004, mut_sigma=0.05, env_var=0.35, sw_period=300.0, rep_free=1.0, rep_fixed=1.0)

def world():
    import jax, jax.numpy as jnp, sim13
    cfg = sim13.Cfg(N=2048, L=1.0, G=64, claim_mode="mass"); init, chunk_one, colour, consts = sim13.make(cfg)
    prm = sim13.params(**BASE); tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    off = {n: (o, sh) for n, sh, o, z in consts["lay"]}
    return cfg, chunk_one, consts, prm, tmpl, off

def load_fork(sd, tmpl):
    z = np.load(os.path.join(FORK, f"fork16_sw_pays_s{sd}_t3000.npz"))
    return {k: np.asarray(z[k]).astype(sd_.dtype) if k in z.files else np.zeros(sd_.shape, sd_.dtype) for k, sd_ in tmpl.items()}

def views(W, off):
    H = off["W_hh"][1][0]
    g = lambda n: W[..., off[n][0]:off[n][0] + int(np.prod(off[n][1]))].reshape(W.shape[:-1] + off[n][1])
    return {n: g(n) for n in off}, H

def implant(st, sd, off):
    """rewire one hidden unit of a random 50% of live agents to: rep > 0 iff sensed E > THR. Returns new state and implanted mask."""
    W = st["W"].copy(); g, H = views(W, off); al = st["alive"]
    rng = np.random.default_rng(1919 + sd); live = np.nonzero(al)[0]; pick = rng.choice(live, size=len(live) // 2, replace=False)
    sthr = np.log1p(THR) - 0.7
    for i in pick:
        k = int(np.argmin(np.abs(g["W_hh"][i]).sum(1) + np.abs(g["W_ho"][i]).sum(1)))
        g["W_ih"][i][:, k] = 0.0; g["W_ih"][i][8, k] = G_IN; g["W_hh"][i][:, k] = 0.0; g["b_h"][i][k] = -G_IN * sthr
        g["W_hh"][i][k, :] = 0.0; g["W_ho"][i][k, :] = 0.0; g["W_ho"][i][:, 3] = 0.0; g["W_ho"][i][k, 3] = W_OUT; g["b_o"][i][3] = 0.0
    for n, (o, sh) in off.items(): W[:, o:o + int(np.prod(sh))] = g[n].reshape(W.shape[0], -1)
    st = dict(st); st["W"] = W.astype(st["W"].dtype); m = np.zeros(len(al), bool); m[pick] = True
    return st, m

def make_chunk(off):
    import jax, jax.numpy as jnp
    cfg, chunk_one, consts, prm, tmpl, _ = world(); step = consts["step"]; N = cfg.N
    oi, (I, H) = off["W_ih"][0], off["W_ih"][1]; oo, (_, A) = off["W_ho"][0], off["W_ho"][1]
    def marker(W):
        wi8 = W[:, oi + 8 * H: oi + 9 * H]; wo3 = W[:, oo + 3: oo + H * A: A]
        return (wi8 * wo3).max(1) > MARK
    z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
    def chunk(s, prm, key, sham):
        def body(c, k):
            al = c["alive"]; m = marker(c["W"]) & al; E = c["E"]
            r = jax.random.uniform(k, (N,)); order = jnp.argsort(jnp.where(al, r, 2.0 + r))
            srt = jnp.argsort(jnp.where(al, 0, 1), stable=True); src = jnp.zeros(N, jnp.int32).at[srt].set(order.astype(jnp.int32))
            eoff = jnp.where((sham > 0) & m, E[src] - E, 0.0)
            c2 = step(c, prm, dict(aclamp=zb, aover=z8, e_off=eoff, rep_use=zb, rep_over=zf))
            rp = (c2["act"][:, 3] > 0).astype(jnp.float32); w0 = al.astype(jnp.float32); wm = m.astype(jnp.float32)
            stats = jnp.stack([f(w) for w in (w0, wm) for f in (lambda w: w.sum(), lambda w: (w * rp).sum(), lambda w: (w * E).sum(),
                                                                    lambda w: (w * rp * E).sum(), lambda w: (w * E * E).sum())])
            return c2, stats
        s, st = jax.lax.scan(body, s, jax.random.split(key, DT))
        return s, st.sum(0)
    return jax.jit(jax.vmap(chunk, in_axes=(0, None, 0, None))), marker, step, chunk_one, prm, tmpl, cfg

def corr(v):
    n, sr, sE, srE, sEE = v
    if n < 3: return float("nan")
    mr, mE = sr / n, sE / n; cov = srE / n - mr * mE; vr = mr - mr * mr; vE = sEE / n - mE * mE
    return float(cov / np.sqrt(vr * vE)) if vr > 1e-12 and vE > 1e-12 else float("nan")

def check():
    import jax, jax.numpy as jnp
    cfg, chunk_one, consts, prm, tmpl, off = world()
    bch, marker, step, _, prm, tmpl, cfg = make_chunk(off)
    out = {}
    s0 = load_fork(0, tmpl); s0, m = implant(s0, 0, off)
    S = {k: jnp.asarray(v)[None] for k, v in s0.items()}
    a, _ = bch(S, prm, jax.random.PRNGKey(5)[None], 0)
    b = jax.jit(lambda s: chunk_one(s, prm, DT))({k: jnp.asarray(v) for k, v in s0.items()})
    out["true_eq_plain_fields_mismatch"] = [k for k in b if not np.array_equal(np.asarray(a[k][0]), np.asarray(b[k]), equal_nan=True)]
    mk = np.asarray(marker(jnp.asarray(s0["W"])) & jnp.asarray(s0["alive"]))
    out["marker_eq_implant"] = bool(np.array_equal(mk, m & s0["alive"])); out["n_implanted"] = int(m.sum()); out["n_live"] = int(s0["alive"].sum())
    c2 = step({k: jnp.asarray(v) for k, v in s0.items()}, prm, dict(aclamp=jnp.zeros(cfg.N, bool), aover=jnp.zeros((cfg.N, 8)), e_off=jnp.zeros(cfg.N), rep_use=jnp.zeros(cfg.N, bool), rep_over=jnp.zeros(cfg.N)))
    rp = np.asarray(c2["act"][:, 3]) > 0; E0 = s0["E"]; sel = m & s0["alive"]
    out["implant_rule_exact_t0"] = bool(np.array_equal(rp[sel], E0[sel] > THR)); out["implant_rule_agree_frac"] = float((rp[sel] == (E0[sel] > THR)).mean())
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "check.json"), "w"), indent=1); print(json.dumps(out), flush=True)

TRAITS = ("m", "a")
def run(arm):
    import jax, jax.numpy as jnp
    path = os.path.join(OUT, f"run19_{arm}_{TAG}.pkl"); ckp = os.path.join(OUT, f"ck19_{arm}_{TAG}.pkl")
    if os.path.exists(path): print("cached", path, flush=True); return
    t0 = time.time(); cfg, chunk_one, consts, prm, tmpl, off = world(); bch, marker, *_ = make_chunk(off); B = len(SEEDS); sham = 1 if arm == "sham" else 0
    if os.path.exists(ckp):
        ck = pickle.load(open(ckp, "rb")); st = {k: jnp.asarray(x) for k, x in ck["st"].items()}; ser, prev, t_start = ck["series"], ck["prev"], ck["t"]
        print(arm, TAG, "resumed at", t_start, flush=True)
    else:
        ss = []
        for sd in SEEDS:
            s, _ = implant(load_fork(sd, tmpl), sd, off); al = s["alive"]; s["anc"] = np.where(al, s["uid"], -1).astype(np.int32); ss.append(s)
        st = {k: jnp.asarray(np.stack([s[k] for s in ss])) for k in tmpl}
        ser = [dict(t=[], pop=[], share=[], stats=[], n=[], n1=[], **{f"{q}_{c}": [] for q in TRAITS for c in ("zbar", "zbar1", "D", "S", "T", "W", "I", "resid")}) for _ in range(B)]
        prev = [None] * B; t_start = 0
        mk = np.asarray(jax.vmap(marker)(st["W"])) & np.asarray(st["alive"])
        for b in range(B): ser[b]["t"].append(0); ser[b]["pop"].append(int(np.asarray(st["alive"][b]).sum())); ser[b]["share"].append(float(mk[b].sum() / max(np.asarray(st["alive"][b]).sum(), 1)))
    pkeys = [jax.random.PRNGKey(191919 + sd) for sd in SEEDS]
    g = lambda k: np.asarray(st[k])
    for t in range(t_start + DT, STEPS + 1, DT):
        st, stats = bch(st, prm, jnp.stack([jax.random.fold_in(k, t) for k in pkeys]), sham); stats = np.asarray(stats, np.float64)
        al, uid, l0n, anc, gpar = g("alive"), g("uid"), g("l0n"), g("anc").astype(np.int64), g("gpar")
        mk = np.asarray(jax.vmap(marker)(st["W"])); ca = g("c_esr"); anc_new = anc.copy()
        for b in range(B):
            S_ = ser[b]; S_["t"].append(t); S_["pop"].append(int(al[b].sum())); S_["share"].append(float((mk[b] & al[b]).sum() / max(al[b].sum(), 1))); S_["stats"].append(stats[b])
            mem = al[b] & (l0n[b] >= 1); mi = np.nonzero(mem)[0]
            cur = dict(uid=uid[b, mi].astype(np.int64), anc=anc[b, mi], zm=mk[b, mi].astype(np.float64), za=ca[b, mi].astype(np.float64) / l0n[b, mi])
            P_ = prev[b]
            if P_ is not None:
                idx = {int(x): i for i, x in enumerate(P_["uid"])}; n, n1 = len(P_["uid"]), len(cur["uid"])
                pi = np.array([idx.get(int(x), -1) for x in cur["anc"]], dtype=np.int64)
                is_s = (cur["anc"] == cur["uid"]) & (pi >= 0); is_o = (~is_s) & (pi >= 0); is_i = pi < 0
                w = np.zeros(n); np.add.at(w, pi[~is_i], 1.0); S_["n"].append(n); S_["n1"].append(n1)
                for q in TRAITS:
                    zp = P_["z" + q]; zc = cur["z" + q]; zbar = zp.mean() if n else 0.0; zbar1 = zc.mean() if n1 else 0.0
                    Sx = (np.sum(w * zp) - w.mean() * n * zbar) / n1; Tx = np.sum(zc[is_o] - zp[pi[is_o]]) / n1
                    Wx = np.sum(zc[is_s] - zp[pi[is_s]]) / n1; Ix = np.sum(zc[is_i] - zbar) / n1; D = zbar1 - zbar
                    for k, val in (("zbar", zbar), ("zbar1", zbar1), ("D", D), ("S", Sx), ("T", Tx), ("W", Wx), ("I", Ix), ("resid", D - (Sx + Tx + Wx + Ix))):
                        S_[f"{q}_{k}"].append(float(val))
            prev[b] = cur; memset = set(cur["uid"].tolist())
            anc_new[b] = -1; anc_new[b, mi] = uid[b, mi]; nm = np.nonzero(al[b] & ~mem)[0]
            anc_new[b, nm] = np.where(np.isin(gpar[b, nm], list(memset)), gpar[b, nm], -1)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t % CK == 0 and t < STEPS:
            pickle.dump(dict(t=t, st={k: np.asarray(x) for k, x in st.items()}, series=ser, prev=prev), open(ckp + ".tmp", "wb"), protocol=5); os.replace(ckp + ".tmp", ckp)
            print(arm, TAG, t, f"{time.time()-t0:.0f}s ckpt", flush=True)
    if os.environ.get("EXP19_FORKDIR"):   # Exp 20 hook: save the final state per seed (no dynamics change)
        os.makedirs(os.environ["EXP19_FORKDIR"], exist_ok=True)
        for b, sd in enumerate(SEEDS): np.savez(os.path.join(os.environ["EXP19_FORKDIR"], f"fork20_s{sd}_t{STEPS}.npz"), **{k: np.asarray(x[b]) for k, x in st.items()})
    res = dict(arm=arm, seeds=SEEDS, steps=STEPS, DT=DT, params=BASE, series=[{k: np.asarray(v) for k, v in s.items()} for s in ser], wall_s=time.time() - t0)
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    if os.path.exists(ckp): os.remove(ckp)
    print("DONE", arm, TAG, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "--check": check()
    else:
        for arm in sys.argv[1:]: run(arm)
