"""Experiment 15: does any selection pressure for interoception exist? (NIGHTLY_LOG 2026-11-15 next step)
sim11 (= sim9c + satiety/gorging-cost mechanic + energy-sensor blind switch, see sim11.py header). World, params and harness are
Exp 11-13's (switching world sw_period=300, Exp 5 base, N=2048, G=64, PRNGKey(1000*s+7), 10-step chunks, guard 0), 3000 steps,
seeds 0-8 as three vmapped batches of 3 (EXP15_SEEDS="0,1,2"/"3,4,5"/"6,7,8"). Arms:
  sw_full        sat_k 0   control (mechanic off == sim9 sw_full, checked bit for bit)
  sw_pays        sat_k 1   eating when well-fed costs energy: sensing energy and eating less when fed wins
  sw_pays_blind  sat_k 1, blind 1   same mechanic, energy sense replaced by a constant for the whole evolution
Exact Price decomposition (as exp13_run.py; residual logged) of three per-agent traits (lifetime means since birth):
  a = c_es/l0n  (energy-sensitivity of the policy: mean d eat / d sensed energy; negative = eats less when fed)
  b = c_esa/l0n (|d eat / d sensed energy|)
  c = c_net/l0n (fitness proxy: net lifetime intake per step, got - gorging penalty)
TONIGHT (2026-11-16) the mechanic check (--paycheck) showed the sat_k=1 mechanic does not reward energy-conditioned eating, so only sw_full and
sw_full_shuffled (heredity broken, drift baseline) were run (--jobs sw_full,sw_full_shuffled); see NIGHTLY_LOG 2026-11-16.
Forks of the full state at t=2500 and 3000 (runs15/fork15_*.npz, NOT committed). Checkpoints every 500 steps; resume bit-for-bit
(same code path as exp13_run.py, tested there).
  python exp15_run.py --bitcheck ; python exp15_run.py --paycheck ; EXP15_SEEDS=0,1,2 python exp15_run.py --procs 2
"""
from __future__ import annotations
import argparse, os, sys, time, json, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP15_OUT", os.path.join(HERE, "runs15"))
STEPS = int(os.environ.get("EXP15_STEPS", "3000")); CK = int(os.environ.get("EXP15_CK", "500")); DT = 10
FORKS = (2500, 3000)
SEEDS = [int(x) for x in os.environ.get("EXP15_SEEDS", "0,1,2").split(",")]
TAG = "s" + "".join(map(str, SEEDS))
BASE = dict(c_basal=0.004, mut_sigma=0.05, env_var=0.35)
SW = 300.0
VARIANTS = dict(sw_full=dict(sw_period=SW), sw_full_shuffled=dict(sw_period=SW, shuffle=1.0), sw_pays=dict(sw_period=SW, sat_k=1.0), sw_pays_blind=dict(sw_period=SW, sat_k=1.0, blind=1.0))
CFGKW = dict(claim_mode="mass")
TRAITS = dict(a="c_es", b="c_esa", c="c_net")
KEYS_TS = ("t", "pop", "eat", "E", "intake", "sat", "es", "eat_E_corr", "net")

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def make(simmod, v, seeds, **prmkw):
    import jax, jax.numpy as jnp
    cfg = simmod.Cfg(N=2048, L=1.0, G=64, **CFGKW)
    binit, bchunk, bcol, consts = simmod.batched(cfg)
    pv = dict(BASE, **VARIANTS[v], **prmkw)
    if simmod.__name__ in ("sim9", "sim9c"): pv.pop("sat_k", None); pv.pop("blind", None)
    prm = simmod.stack_params([simmod.params(**pv) for _ in seeds])
    keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for s in seeds])
    return cfg, binit(keys, prm), bchunk, prm

def bitcheck(steps=300, seeds=(0, 1, 2, 3, 4)):
    """mechanic OFF (sat_k=0, blind=0) must reproduce sim9 (and sim9c) on every sim9 field, bit for bit; mechanic ON must differ."""
    setup()
    import sim9, sim9c, sim11
    out = {}
    for sd in seeds:
        r = {}
        _, s9, c9, p9 = make(sim9, "sw_full", [sd]); _, s9c, c9c, p9c = make(sim9c, "sw_full", [sd]); _, s11, c11, p11 = make(sim11, "sw_full", [sd])
        for _ in range(steps // DT):
            s9 = c9(s9, p9, DT); s9c = c9c(s9c, p9c, DT); s11 = c11(s11, p11, DT)
        m9 = [k for k in s9 if not np.array_equal(np.asarray(s9[k]), np.asarray(s11[k]), equal_nan=True)]
        m9c = [k for k in s9c if not np.array_equal(np.asarray(s9c[k]), np.asarray(s11[k]), equal_nan=True)]
        r = dict(n_fields_sim9=len(s9), n_fields_sim11=len(s11), mismatched_vs_sim9=m9, mismatched_vs_sim9c=m9c,
                 new_obs_zero=bool(all(np.all(np.asarray(s11[k]) == 0) for k in ("satstep", "c_sat"))))
        out[f"seed{sd}"] = r
        print(sd, r, flush=True)
    _, sp, cp, pp = make(sim11, "sw_pays", [0]); _, sf, cf, pf = make(sim11, "sw_full", [0]); _, sb, cb, pb = make(sim11, "sw_pays_blind", [0])
    for _ in range(steps // DT):
        sp = cp(sp, pp, DT); sf = cf(sf, pf, DT); sb = cb(sb, pb, DT)
    out["mechanic_on_changes_dynamics"] = not np.array_equal(np.asarray(sp["E"]), np.asarray(sf["E"]))
    out["blind_changes_dynamics_vs_pays"] = not np.array_equal(np.asarray(sb["E"]), np.asarray(sp["E"]))
    al = np.asarray(sp["alive"]); out["mean_satpen_pays_300"] = float(np.asarray(sp["satstep"])[al].mean())
    out["bit_for_bit"] = all(not r["mismatched_vs_sim9"] and not r["mismatched_vs_sim9c"] for k, r in out.items() if k.startswith("seed"))
    out["steps"] = steps; out["seeds"] = list(seeds)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "bitcheck.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("seed")}), flush=True); return out

def run(v, steps=STEPS, out=OUT, ck_every=CK, stop_at=None, forks=FORKS, prefix="run15"):
    setup()
    import jax, jax.numpy as jnp, sim11
    path = os.path.join(out, f"{prefix}_{v}_{TAG}.pkl"); ckp = os.path.join(out, f"ck15_{v}_{TAG}.pkl")
    if os.path.exists(path): return v, 0.0, "cached"
    t0 = time.time(); B = len(SEEDS)
    cfg, st, bchunk, prm = make(sim11, v, SEEDS)
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
            act = g("act"); inta = g("intake"); En = g("E"); sat = g("satstep"); es = g("esstep"); cn = g("cstep")
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
                eat = (act[b, ia, 2] + 1) / 2; Eb = En[b, ia]
                cc = float(np.corrcoef(eat, Eb)[0, 1]) if eat.std() > 1e-9 and Eb.std() > 1e-9 else float("nan")
                for k, val in (("t", t), ("pop", int(len(ia))), ("eat", float(eat.mean())), ("E", float(Eb.mean())), ("intake", float(inta[b, ia].mean())),
                               ("sat", float(sat[b, ia].mean())), ("es", float(es[b, ia].mean())), ("eat_E_corr", cc), ("net", float((inta[b, ia] - sat[b, ia]).mean()))):
                    S_[k].append(val)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t in forks:
            for b, s in enumerate(SEEDS):
                d = {k: np.asarray(x[b]) for k, x in st.items()}
                tmp = os.path.join(out, f"fork15_{v}_s{s}_t{t}.tmp.npz"); np.savez_compressed(tmp, **d); os.replace(tmp, os.path.join(out, f"fork15_{v}_s{s}_t{t}.npz"))
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

SATKS = (1.0, 3.0); THRS = (1.0, 1.3, 1.6)
def paycheck(steps=400, seeds=(0, 1, 2)):
    """Hand-coded eating policies on the sat_k=1 world (brains from the random initial population drive everything else):
    aware   eat = 1 if E < e_thr else 0 (reads true E)       blind  constant eat = the aware run's mean eat (same average food-seeking, no E)
    always  eat = 1                                           off    mechanic off, always eat
    Measured over the last 200 steps: mean population, mean E, mean gorging penalty, mean net intake (got - pen) per live agent-step."""
    setup()
    import jax, jax.numpy as jnp, sim11
    cfg = sim11.Cfg(N=2048, L=1.0, G=64, **CFGKW)
    init, chunk_one, colour, consts = sim11.make(cfg); step = consts["step"]
    N = cfg.N; z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
    def sim(seed, pol, sat_k, eat_c, e_thr):
        prm = sim11.params(**dict(BASE, sw_period=SW, sat_k=sat_k)); s = init(jax.random.PRNGKey(1000 * seed + 7), prm)
        def body(c, _):
            eo = jnp.where(c["E"] < e_thr, 1.0, 0.0) if pol == "aware" else jnp.full(N, eat_c)
            c2 = step(c, prm, dict(aclamp=zb, aover=z8, e_off=zf, eat_use=jnp.ones(N, bool), eat_over=eo))
            al = c2["alive"]; n = jnp.maximum(al.sum(), 1)
            return c2, dict(pop=al.sum(), E=(c2["E"] * al).sum() / n, sat=(c2["satstep"] * al).sum() / n, net=((c2["intake"] - c2["satstep"]) * al).sum() / n, eat=(eo * al).sum() / n)
        _, r = jax.lax.scan(body, s, None, length=steps)
        return {k: float(np.asarray(v)[-200:].mean()) for k, v in r.items()}
    out = {}
    for sk in SATKS:
        for thr in THRS:
            rows = []
            for sd in seeds:
                a = sim(sd, "aware", sk, 0.0, thr); m = a["eat"]
                rows.append(dict(aware=a, blind_matched=sim(sd, "const", sk, m, thr), always=sim(sd, "const", sk, 1.0, thr), off=sim(sd, "const", 0.0, 1.0, thr)))
                print(sk, thr, sd, json.dumps(rows[-1]), flush=True)
            out[f"sat{sk}_thr{thr}"] = {pol: {k: float(np.mean([r[pol][k] for r in rows])) for k in rows[0][pol]} for pol in rows[0]}
            out[f"sat{sk}_thr{thr}"]["aware_minus_blind_net"] = float(np.mean([r["aware"]["net"] - r["blind_matched"]["net"] for r in rows]))
            out[f"sat{sk}_thr{thr}"]["aware_minus_blind_pop"] = float(np.mean([r["aware"]["pop"] - r["blind_matched"]["pop"] for r in rows]))
            out[f"sat{sk}_thr{thr}"]["seeds"] = list(seeds)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "paycheck.json"), "w"), indent=1); print(json.dumps(out), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); ap.add_argument("--bitcheck", action="store_true"); ap.add_argument("--paycheck", action="store_true")
    ap.add_argument("--jobs", default=None, help="comma list of variants (default all)")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.bitcheck: bitcheck(); sys.exit(0)
    if a.paycheck: paycheck(); sys.exit(0)
    t0 = time.time(); vs = a.jobs.split(",") if a.jobs else list(VARIANTS)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for v, w, d in pool.imap_unordered(run, vs):
            print("DONE", v, TAG, f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
