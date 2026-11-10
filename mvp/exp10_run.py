"""Experiment 10a: wall-protected intake (sim8 `guard`), the "make walls privately valuable" test (option (b), Exp 9 log).
Switching world (sw_period=300, Exp 5 base). 6 arms x seeds 0-2, 3000 steps, same harness as Exp 5/9 (3 seeds vmapped,
PRNGKey(1000*s+7), 10-step chunks):
  sw_full             guard 0   control (== Exp 5 sw_full, checked bit for bit against runs9 fork at 2500)
  sw_guard            guard 1   walls you built protect your harvest (priority + F-immune) and your energy from theft
  sw_guard_inert      guard 1, build_eff 0   walls give private claims but no shelter (no F damping, no drag)
  sw_guard_shuffled   guard 1, shuffle 1     heredity broken (selection cannot accumulate a building strategy)
  sw_pecost_pe        guard 0   Exp 5 main arm (control)
  sw_pecost_pe_guard  guard 1   main arm + guard
Exact Price decomposition (same identity as exp5_price.py, D = S+T+W+I; residual logged) of trait
  z = lifetime mean build action since birth (c_build / l0n), and of z2 = lifetime mean claim (c_claim / l0n).
Fitness = descendants at next 10-step boundary. Fork saved at t=2500 (runs10/fork10_*.npz, NOT committed).
  python exp10_run.py --bitcheck ; python exp10_run.py --procs 6
"""
from __future__ import annotations
import argparse, os, sys, time, json, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP10_OUT", os.path.join(HERE, "runs10"))
STEPS = 3000; FORK_T = 2500; DT = 10
SEEDS = [0, 1, 2]
BASE = dict(c_basal=0.004, mut_sigma=0.05, env_var=0.35)
SW = 300.0; PEC = 0.01
VARIANTS = dict(sw_full=dict(sw_period=SW), sw_guard=dict(sw_period=SW, guard=1.0),
                sw_guard_inert=dict(sw_period=SW, guard=1.0, build_eff=0.0),
                sw_guard_shuffled=dict(sw_period=SW, guard=1.0, shuffle=1.0),
                sw_pecost_pe=dict(sw_period=SW, pe_cost=PEC, pe_sense=1.0),
                sw_pecost_pe_guard=dict(sw_period=SW, pe_cost=PEC, pe_sense=1.0, guard=1.0))

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def make(simmod, v, seeds, **cfgkw):
    import jax, jax.numpy as jnp
    cfg = simmod.Cfg(N=2048, L=1.0, G=64, **cfgkw)
    binit, bchunk, bcol, consts = simmod.batched(cfg)
    pv = dict(BASE, **VARIANTS[v])
    if simmod.__name__ != "sim8": pv.pop("guard", None)
    prm = simmod.stack_params([simmod.params(**pv) for _ in seeds])
    keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for s in seeds])
    return cfg, binit(keys, prm), bchunk, prm

def bitcheck(steps=300):
    setup()
    import sim7, sim8
    out = {}
    for v in ("sw_full", "sw_pecost_pe"):
        _, s7, c7, p7 = make(sim7, v, SEEDS, attr=True)
        _, s8, c8, p8 = make(sim8, v, SEEDS)
        for _ in range(steps // DT):
            s7 = c7(s7, p7, DT); s8 = c8(s8, p8, DT)
        mism = [k for k in s7 if not np.array_equal(np.asarray(s7[k]), np.asarray(s8[k]), equal_nan=True)]
        out[v] = dict(n_fields=len(s7), mismatched=mism, bit_for_bit=len(mism) == 0)
    # guard=1 must actually change the dynamics
    _, s8g, c8g, p8g = make(sim8, "sw_guard", SEEDS); _, s8f, c8f, p8f = make(sim8, "sw_full", SEEDS)
    for _ in range(steps // DT):
        s8g = c8g(s8g, p8g, DT); s8f = c8f(s8f, p8f, DT)
    out["guard_changes_dynamics"] = not np.array_equal(np.asarray(s8g["E"]), np.asarray(s8f["E"]))
    out["mean_claim_guard_300"] = float(np.asarray(s8g["clstep"])[np.asarray(s8g["alive"])].mean())
    out["steps"] = steps
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "bitcheck.json"), "w"), indent=1)
    print(json.dumps(out), flush=True); return out

TRAITS = dict(a="c_build", b="c_claim")

def worker(v):
    setup()
    import jax, jax.numpy as jnp, sim8
    path = os.path.join(OUT, f"run10_{v}.pkl")
    if os.path.exists(path) and all(os.path.exists(os.path.join(OUT, f"fork10_{v}_s{s}.npz")) for s in SEEDS):
        return v, 0.0, "cached"
    t0 = time.time(); B = len(SEEDS)
    cfg, st, bchunk, prm = make(sim8, v, SEEDS)
    keys_ts = ("t", "pop", "build", "claim", "intake", "E", "wall", "live_frac", "own_share_here", "pe")
    series = [dict(**{k: [] for k in keys_ts}, n=[], n1=[], **{f"{q}_{c}": [] for q in TRAITS for c in ("zbar", "zbar1", "D", "S", "T", "W", "I", "resid")}) for _ in range(B)]
    prev = [None] * B
    g = lambda k: np.asarray(st[k])
    al = g("alive"); st = dict(st); st["anc"] = jnp.asarray(np.where(al, g("uid"), -1).astype(np.int32))
    cs = cfg.L / cfg.G
    for t in range(DT, STEPS + 1, DT):
        st = bchunk(st, prm, DT)
        al, uid, l0n, anc, gpar = g("alive"), g("uid"), g("l0n"), g("anc").astype(np.int64), g("gpar")
        tr = {q: g(k) for q, k in TRAITS.items()}
        anc_new = anc.copy()
        if t % 50 == 0:
            Wl = g("Wl"); Ad = g("Adep"); pos = g("pos"); act = g("act"); cl = g("clstep"); inta = g("intake"); E = g("E"); pe = g("pestep")
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
                               ("intake", float(inta[b, ia].mean())), ("E", float(E[b, ia].mean())), ("wall", float(Wl[b].sum())),
                               ("live_frac", float(Ad[b].sum() / max(Wl[b].sum(), 1e-12))), ("own_share_here", float(np.mean(oh / np.maximum(Wh, 1e-6)))),
                               ("pe", float(pe[b, ia].mean()))):
                    S_[k].append(val)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t == FORK_T:
            for b, s in enumerate(SEEDS):
                d = {k: np.asarray(x[b]) for k, x in st.items()}
                tmp = os.path.join(OUT, f"fork10_{v}_s{s}.tmp.npz"); np.savez_compressed(tmp, **d); os.replace(tmp, os.path.join(OUT, f"fork10_{v}_s{s}.npz"))
        if t % 500 == 0:
            print(v, t, f"{time.time()-t0:.0f}s", flush=True)
    out = dict(variant=v, params=dict(BASE, **VARIANTS[v]), seeds=SEEDS, DT=DT, series=[{k: np.asarray(x) for k, x in s.items()} for s in series], wall_s=time.time() - t0)
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return v, out["wall_s"], "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--bitcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--only", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.bitcheck:
        bitcheck(a.steps); sys.exit(0)
    V = [a.only] if a.only else list(VARIANTS)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for v, w, d in pool.imap_unordered(worker, V):
            print("DONE", v, f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
