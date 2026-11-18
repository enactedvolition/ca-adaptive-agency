"""Experiment 16: a world where sensing energy pays (NIGHTLY_LOG 2026-11-18). sim12 = sim11 + brain-gated reproduction (rep_free:
simulator energy gate 1.6 -> E_rep_lo 0.35, so the brain must time reproduction by its own energy; see sim12.py header).
World, params and harness are Exp 11-15's (switching world sw_period=300, N=2048, G=64, PRNGKey(1000*s+7), 10-step chunks, guard 0),
3000 steps, seeds 0-8 as three vmapped batches of 3 (EXP16_SEEDS). Arms:
  sw_full        rep_free 0             control (== sim9/sim11 sw_full, checked bit for bit)
  sw_pays        rep_free 1             brain-gated reproduction
  sw_pays_blind  rep_free 1, blind 1    same, energy sense replaced by a constant for the whole evolution
Exact Price decomposition (residual logged) of per-agent lifetime means since birth:
  a = c_esr/l0n (d rep / d sensed energy; positive = reproduces more when fed)   b = c_es/l0n (d eat / d sensed energy)   c = c_net/l0n
Forks of the full state at t=2500 and 3000 (runs16/fork16_*.npz, NOT committed). Checkpoints every 500 steps; resume bit-for-bit.
  python exp16_run.py --bitcheck ; python exp16_run.py --paycheck ; EXP16_SEEDS=0,1,2 python exp16_run.py --procs 2
"""
from __future__ import annotations
import argparse, os, sys, time, json, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP16_OUT", os.path.join(HERE, "runs16"))
STEPS = int(os.environ.get("EXP16_STEPS", "3000")); CK = int(os.environ.get("EXP16_CK", "500")); DT = 10
FORKS = (2500, 3000)
SEEDS = [int(x) for x in os.environ.get("EXP16_SEEDS", "0,1,2").split(",")]
TAG = "s" + "".join(map(str, SEEDS))
BASE = dict(c_basal=0.004, mut_sigma=0.05, env_var=0.35)
SW = 300.0
VARIANTS = dict(sw_full=dict(sw_period=SW), sw_pays=dict(sw_period=SW, rep_free=1.0), sw_pays_blind=dict(sw_period=SW, rep_free=1.0, blind=1.0))
CFGKW = dict(claim_mode="mass")
TRAITS = dict(a="c_esr", b="c_es", c="c_net")
KEYS_TS = ("t", "pop", "eat", "E", "intake", "esr", "es", "rep_E_corr", "net", "rep", "births")

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def make(simmod, v, seeds, **prmkw):
    import jax, jax.numpy as jnp
    cfg = simmod.Cfg(N=2048, L=1.0, G=64, **CFGKW)
    binit, bchunk, bcol, consts = simmod.batched(cfg)
    pv = dict(BASE, **VARIANTS[v], **prmkw)
    if simmod.__name__ in ("sim9", "sim9c", "sim11"): pv.pop("rep_free", None)
    if simmod.__name__ in ("sim9", "sim9c"): pv.pop("sat_k", None); pv.pop("blind", None)
    prm = simmod.stack_params([simmod.params(**pv) for _ in seeds])
    keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for s in seeds])
    return cfg, binit(keys, prm), bchunk, prm

def bitcheck(steps=300, seeds=(0, 1, 2, 3, 4)):
    """mechanic OFF (rep_free=0) must reproduce sim11 on every sim11 field and sim9 on every sim9 field, bit for bit; ON must differ."""
    setup()
    import sim9, sim11, sim12
    out = {}
    for sd in seeds:
        _, s9, c9, p9 = make(sim9, "sw_full", [sd]); _, s11, c11, p11 = make(sim11, "sw_full", [sd]); _, s12, c12, p12 = make(sim12, "sw_full", [sd])
        for _ in range(steps // DT):
            s9 = c9(s9, p9, DT); s11 = c11(s11, p11, DT); s12 = c12(s12, p12, DT)
        m9 = [k for k in s9 if not np.array_equal(np.asarray(s9[k]), np.asarray(s12[k]), equal_nan=True)]
        m11 = [k for k in s11 if not np.array_equal(np.asarray(s11[k]), np.asarray(s12[k]), equal_nan=True)]
        r = dict(n_fields_sim9=len(s9), n_fields_sim11=len(s11), n_fields_sim12=len(s12), mismatched_vs_sim9=m9, mismatched_vs_sim11=m11,
                 births_logged=float(np.asarray(s12["c_rep"]).sum()))
        out[f"seed{sd}"] = r
        print(sd, r, flush=True)
    _, sp, cp, pp = make(sim12, "sw_pays", [0]); _, sf, cf, pf = make(sim12, "sw_full", [0]); _, sb, cb, pb = make(sim12, "sw_pays_blind", [0])
    for _ in range(steps // DT):
        sp = cp(sp, pp, DT); sf = cf(sf, pf, DT); sb = cb(sb, pb, DT)
    out["mechanic_on_changes_dynamics"] = not np.array_equal(np.asarray(sp["E"]), np.asarray(sf["E"]))
    out["blind_changes_dynamics_vs_pays"] = not np.array_equal(np.asarray(sb["E"]), np.asarray(sp["E"]))
    out["pop300"] = dict(sw_full=int(np.asarray(sf["alive"]).sum()), sw_pays=int(np.asarray(sp["alive"]).sum()), sw_pays_blind=int(np.asarray(sb["alive"]).sum()))
    out["bit_for_bit"] = all(not r["mismatched_vs_sim9"] and not r["mismatched_vs_sim11"] for k, r in out.items() if k.startswith("seed"))
    out["steps"] = steps; out["seeds"] = list(seeds)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "bitcheck.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("seed")}), flush=True); return out

def run(v, steps=STEPS, out=OUT, ck_every=CK, stop_at=None, forks=FORKS, prefix="run16"):
    setup()
    import jax, jax.numpy as jnp, sim12
    path = os.path.join(out, f"{prefix}_{v}_{TAG}.pkl"); ckp = os.path.join(out, f"ck16_{v}_{TAG}.pkl")
    if os.path.exists(path): return v, 0.0, "cached"
    t0 = time.time(); B = len(SEEDS)
    cfg, st, bchunk, prm = make(sim12, v, SEEDS)
    if os.path.exists(ckp):
        ck = pickle.load(open(ckp, "rb"))
        st = {k: jnp.asarray(x) for k, x in ck["st"].items()}; series, prev, t_start = ck["series"], ck["prev"], ck["t"]
        print(v, "resumed at", t_start, flush=True)
    else:
        series = [dict(**{k: [] for k in KEYS_TS}, n=[], n1=[], **{f"{q}_{c}": [] for q in TRAITS for c in ("zbar", "zbar1", "D", "S", "T", "W", "I", "resid")}) for _ in range(B)]
        prev = [None] * B; t_start = 0
        al = np.asarray(st["alive"]); st = dict(st); st["anc"] = jnp.asarray(np.where(al, np.asarray(st["uid"]), -1).astype(np.int32))
    g = lambda k: np.asarray(st[k])
    for t in range(t_start + DT, steps + 1, DT):
        st = bchunk(st, prm, DT)
        al, uid, l0n, anc, gpar = g("alive"), g("uid"), g("l0n"), g("anc").astype(np.int64), g("gpar")
        tr = {q: g(k) for q, k in TRAITS.items()}
        anc_new = anc.copy()
        if t % 50 == 0:
            act = g("act"); inta = g("intake"); En = g("E"); sat = g("satstep"); es = g("esstep"); esr = g("esrstep"); rps = g("repstep")
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
                ia = np.nonzero(al[b])[0]
                eat = (act[b, ia, 2] + 1) / 2; Eb = En[b, ia]; rp = (act[b, ia, 3] > 0).astype(float)
                cc = float(np.corrcoef(rp, Eb)[0, 1]) if rp.std() > 1e-9 and Eb.std() > 1e-9 else float("nan")
                bi = rps[b] > 0
                for k, val in (("t", t), ("pop", int(len(ia))), ("eat", float(eat.mean())), ("E", float(Eb.mean())), ("intake", float(inta[b, ia].mean())),
                               ("esr", float(esr[b, ia].mean())), ("es", float(es[b, ia].mean())), ("rep_E_corr", cc), ("net", float((inta[b, ia] - sat[b, ia]).mean())),
                               ("rep", float(rp.mean())), ("births", int(bi.sum()))):
                    S_[k].append(val)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t in forks:
            for b, s in enumerate(SEEDS):
                d = {k: np.asarray(x[b]) for k, x in st.items()}
                tmp = os.path.join(out, f"fork16_{v}_s{s}_t{t}.tmp.npz"); np.savez_compressed(tmp, **d); os.replace(tmp, os.path.join(out, f"fork16_{v}_s{s}_t{t}.npz"))
        if t % ck_every == 0 and t < steps:
            ck = dict(t=t, st={k: np.asarray(x) for k, x in st.items()}, series=series, prev=prev)
            pickle.dump(ck, open(ckp + ".tmp", "wb"), protocol=5); os.replace(ckp + ".tmp", ckp)
            print(v, TAG, t, f"{time.time()-t0:.0f}s ckpt", flush=True)
            if stop_at is not None and t >= stop_at: return v, time.time() - t0, "stopped"
    res = dict(variant=v, params=dict(BASE, **VARIANTS[v]), seeds=SEEDS, DT=DT, steps=steps,
               series=[{k: np.asarray(x) for k, x in s.items()} for s in series], wall_s=time.time() - t0)
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    if os.path.exists(ckp): os.remove(ckp)
    return v, res["wall_s"], "ok"

THRS = (0.8, 1.2, 1.6)
FORK15 = os.path.join(HERE, "runs15")
def paycheck(steps=400, seeds=(0, 1, 2), start="fork"):
    """Hand-coded rep policies on the rep_free=1 world; brains drive every other action.
    start="fork": Exp 15b sw_full t=3000 forks (pre-registered primary); start="init": random initial population (Exp 15 protocol).
      aware  rep = +1 iff true E > thr (thr in THRS)    blind  rep = +1 with prob p = the aware run's mean rep indicator (last 200 steps)
      brain  rep from the brain, rep_free = 1           off    rep from the brain, rep_free = 0 (the old world)
    Measured over the last 200 steps: mean population, mean E, mean net intake per live agent-step, births per step, mean rep indicator."""
    setup()
    import jax, jax.numpy as jnp, sim12
    cfg = sim12.Cfg(N=2048, L=1.0, G=64, **CFGKW)
    init, chunk_one, colour, consts = sim12.make(cfg); step = consts["step"]
    N = cfg.N; z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
    def state0(seed, prm):
        if start == "init": return init(jax.random.PRNGKey(1000 * seed + 7), prm)
        tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm); z = np.load(os.path.join(FORK15, f"fork15_sw_full_s{seed}_t3000.npz"))
        st = {}
        for kk, sd in tmpl.items():
            st[kk] = jnp.asarray(np.asarray(z[kk]).astype(sd.dtype)) if kk in z.files else jnp.zeros(sd.shape, sd.dtype)
        return st
    @jax.jit
    def simj(s, prm, mode, thr, p, pkey):
        def body(c, k):
            c, kk = c, k
            rnd = jax.random.uniform(kk, (N,))
            ro = jnp.where(mode == 0, jnp.where(c["E"] > thr, 1.0, -1.0), jnp.where(rnd < p, 1.0, -1.0))
            ctl = dict(aclamp=zb, aover=z8, e_off=zf, rep_use=jnp.full(N, mode < 2), rep_over=ro)
            c2 = step(c, prm, ctl)
            al = c2["alive"]; n = jnp.maximum(al.sum(), 1)
            repind = jnp.where(mode < 2, (ro > 0), c2["act"][:, 3] > 0)
            return c2, dict(pop=al.sum(), E=(c2["E"] * al).sum() / n, net=((c2["intake"] - c2["satstep"]) * al).sum() / n,
                            births=c2["repstep"].sum(), rep=(repind * c["alive"]).sum() / jnp.maximum(c["alive"].sum(), 1))
        _, r = jax.lax.scan(body, s, jax.random.split(pkey, steps))
        return r
    def sim(seed, mode, thr=0.0, p=0.0, rf=1.0):
        prm = sim12.params(**dict(BASE, sw_period=SW, rep_free=rf)); s = state0(seed, prm)
        r = simj(s, prm, jnp.int32(mode), jnp.float32(thr), jnp.float32(p), jax.random.PRNGKey(424242 + seed))
        return {k: float(np.asarray(v)[-200:].mean()) for k, v in r.items()}
    out = dict(start=start, steps=steps, seeds=list(seeds), thrs=list(THRS), E_rep_lo=cfg.E_rep_lo, per={})
    for sd in seeds:
        d = dict(brain=sim(sd, 2), off=sim(sd, 2, rf=0.0))
        for thr in THRS:
            a = sim(sd, 0, thr=thr); d[f"aware{thr}"] = a; d[f"blind{thr}"] = sim(sd, 1, p=a["rep"])
        out["per"][f"s{sd}"] = d; print(start, sd, json.dumps(d), flush=True)
    summ = {}
    for thr in THRS:
        dp = [(out["per"][f"s{sd}"][f"aware{thr}"]["pop"] / out["per"][f"s{sd}"][f"blind{thr}"]["pop"] - 1) for sd in seeds]
        dn = [(out["per"][f"s{sd}"][f"aware{thr}"]["net"] / out["per"][f"s{sd}"][f"blind{thr}"]["net"] - 1) for sd in seeds]
        summ[str(thr)] = dict(pop_rel=dp, pop_rel_mean=float(np.mean(dp)), net_rel=dn, net_rel_mean=float(np.mean(dn)),
                              pop_aware=float(np.mean([out["per"][f"s{sd}"][f"aware{thr}"]["pop"] for sd in seeds])),
                              pop_blind=float(np.mean([out["per"][f"s{sd}"][f"blind{thr}"]["pop"] for sd in seeds])))
    best = max(THRS, key=lambda t: summ[str(t)]["pop_aware"])
    b = summ[str(best)]
    out["summary"] = summ; out["best_thr"] = best
    out["PASS_prereg"] = bool(b["pop_rel_mean"] >= 0.05 and all(x > 0 for x in b["pop_rel"]) and b["net_rel_mean"] >= 0)
    out["PASS_literal_suggested"] = bool(b["net_rel_mean"] >= 0.05 and all(x > 0 for x in b["net_rel"]) and b["pop_rel_mean"] >= 0)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, f"paycheck_{start}.json"), "w"), indent=1)
    print(json.dumps(dict(summary=summ, best=best, PASS_prereg=out["PASS_prereg"], PASS_literal=out["PASS_literal_suggested"])), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); ap.add_argument("--bitcheck", action="store_true"); ap.add_argument("--paycheck", action="store_true")
    ap.add_argument("--jobs", default=None, help="comma list of variants (default all)")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.bitcheck: bitcheck(); sys.exit(0)
    if a.paycheck: paycheck(start=os.environ.get("EXP16_PAYSTART", "fork")); sys.exit(0)
    t0 = time.time(); vs = a.jobs.split(",") if a.jobs else list(VARIANTS)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for v, w, d in pool.imap_unordered(run, vs):
            print("DONE", v, TAG, f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
