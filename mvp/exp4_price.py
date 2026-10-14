"""Experiment 4a ("make regulation pay"): fresh sim4 runs, 3 seeds per variant, exact Price decomposition of mean lambda.

Variants (sim4 params; base = exp2 main: c_basal 0.004, mut 0.05, env_var 0.35):
  full, shuffled, no_cost, static_env   (controls, regulation knobs off)
  pe        : agents sense own prediction error (pe_sense=1)
  lamcost   : instability costs energy (lam_cost=LAMC per unit positive local log-stretch)
  pe_lamcost: both
Price over DT-step intervals, fitness = descendants (incl. self) present at next boundary; z = (a) lambda since birth under
actual inputs (sim-state l0s/l0n), (b) lambda under a standardized bank of real input sequences (from pe seed 0, steps 1001-1040;
channel 15 zeroed for variants that cannot sense PE). D = S + T + W + I exactly.
  python exp4_price.py bank ; python exp4_price.py run --procs 7
"""
from __future__ import annotations
import argparse, os, pickle, sys, time, json
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs4")
DT = 10
STEPS = int(os.environ.get("P4_STEPS", 3000))
FORK_T = int(os.environ.get("P4_FORK", 2500))
BANK_BURN, BANK_T = 10, 40
LAMC = 0.01
BASE = dict(c_basal=0.004, mut_sigma=0.05, env_var=0.35)
VARIANTS = dict(full={}, shuffled=dict(shuffle=1.0), no_cost=dict(cost_mul=0.0), static_env=dict(feedback=0.0),
                pe=dict(pe_sense=1.0), lamcost=dict(lam_cost=LAMC), pe_lamcost=dict(pe_sense=1.0, lam_cost=LAMC))
SEEDS = [0, 1, 2]

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def prm_of(v):
    import sim4
    return sim4.params(**dict(BASE, **VARIANTS[v]))

def make_sim(v, seeds):
    import jax, jax.numpy as jnp, sim4
    cfg = sim4.Cfg(N=2048, L=1.0, G=64)
    binit, bchunk, bcol, consts = sim4.batched(cfg)
    prm = sim4.stack_params([prm_of(v) for _ in seeds])
    keys = jnp.stack([jax.random.PRNGKey(1000 * s + 7) for s in seeds])
    return cfg, binit(keys, prm), bchunk, prm, consts

def bank_lambda_fn(cfg, bank):
    import jax, jax.numpy as jnp, sim4
    lay, P, _ = sim4.layout(cfg.H); H = cfg.H
    bank = jnp.asarray(bank, jnp.float32)
    v0 = jnp.asarray(np.random.default_rng(99).normal(size=H), jnp.float32); v0 = v0 / jnp.linalg.norm(v0)
    def one(W, h0):
        g = {n: W[o:o + z].reshape(sh) for n, sh, o, z in lay}
        def seq(xs):
            def body(c, x):
                h, v = c
                hn = jnp.tanh(x @ g["W_ih"] + g["b_h"] + h @ g["W_hh"])
                jv = (1 - hn**2) * (v @ g["W_hh"]); n = jnp.linalg.norm(jv) + 1e-30
                return (hn, jv / n), jnp.log(n)
            _, ls = jax.lax.scan(body, (h0, v0), xs)
            return ls[BANK_BURN:].mean()
        return jax.vmap(seq)(bank).mean()
    return jax.jit(jax.vmap(jax.vmap(one)))

def run_bank():
    setup()
    cfg, st, bchunk, prm, consts = make_sim("pe", [0])
    for _ in range(100):
        st = bchunk(st, prm, 10)
    X, U = [], []
    for _ in range(BANK_T):
        st = bchunk(st, prm, 1)
        X.append(np.asarray(st["xprev"][0])); U.append(np.asarray(st["uid"][0]) * np.asarray(st["alive"][0]))
    X = np.stack(X); U = np.stack(U)
    ok = np.nonzero((U == U[0]).all(0) & (U[0] > 0))[0]
    pick = np.random.default_rng(0).choice(ok, 4, replace=False)
    bank = X[:, pick].transpose(1, 0, 2)
    np.savez(os.path.join(OUT, "input_bank.npz"), bank=bank, slots=pick, uids=U[0][pick], t0=1001, source="sim4 pe seed 0")
    print("bank", bank.shape, flush=True)

def worker(v):
    setup()
    import jax, jax.numpy as jnp
    path = os.path.join(OUT, f"price_{v}.pkl")
    if os.path.exists(path):
        return path, 0.0, "cached"
    B = len(SEEDS)
    cfg, st, bchunk, prm, consts = make_sim(v, SEEDS)
    N = cfg.N
    bank = np.load(os.path.join(OUT, "input_bank.npz"))["bank"].copy()
    if VARIANTS[v].get("pe_sense", 0) == 0:
        bank[..., 15] = 0.0
    lamb = bank_lambda_fn(cfg, bank)
    series = [dict(t=[], n=[], n1=[], pop=[], mean_E=[], mean_cost=[], **{f"{q}_{c}": [] for q in ("a", "b") for c in ("zbar", "zbar1", "D", "S", "T", "W", "I")})
              for _ in range(B)]
    prev = [None] * B; t0 = time.time()
    g = lambda k: np.asarray(st[k])
    al = g("alive"); st = dict(st); st["anc"] = jnp.asarray(np.where(al, g("uid"), -1).astype(np.int32))
    for t in range(DT, STEPS + 1, DT):
        st = bchunk(st, prm, DT)
        al, uid, l0s, l0n, anc, gpar = g("alive"), g("uid"), g("l0s"), g("l0n"), g("anc").astype(np.int64), g("gpar")
        zb_all = np.asarray(lamb(st["W"], st["h"]))
        E, cst = g("E"), g("cstep")
        anc_new = anc.copy()
        for b in range(B):
            mem = al[b] & (l0n[b] >= 1); mi = np.nonzero(mem)[0]
            cur = dict(uid=uid[b, mi].astype(np.int64), za=l0s[b, mi] / l0n[b, mi], zb=zb_all[b, mi].astype(np.float64), anc=anc[b, mi])
            S_ = series[b]
            P_ = prev[b]
            if P_ is not None:
                idx = {int(x): i for i, x in enumerate(P_["uid"])}
                n, n1 = len(P_["uid"]), len(cur["uid"])
                pi = np.array([idx.get(int(x), -1) for x in cur["anc"]])
                is_s = (cur["anc"] == cur["uid"]) & (pi >= 0); is_o = (~is_s) & (pi >= 0); is_i = pi < 0
                w = np.zeros(n); np.add.at(w, pi[~is_i], 1.0)
                S_["t"].append(t); S_["n"].append(n); S_["n1"].append(n1)
                S_["pop"].append(int(al[b].sum())); S_["mean_E"].append(float(E[b][al[b]].mean())); S_["mean_cost"].append(float(cst[b][al[b]].mean()))
                for q in ("a", "b"):
                    zp = P_["z" + q]; zc = cur["z" + q]; zbar = zp.mean(); zbar1 = zc.mean()
                    Sx = (np.sum(w * zp) - w.mean() * n * zbar) / n1
                    Tx = np.sum(zc[is_o] - zp[pi[is_o]]) / n1
                    Wx = np.sum(zc[is_s] - zp[pi[is_s]]) / n1
                    Ix = np.sum(zc[is_i] - zbar) / n1
                    for k, val in (("zbar", zbar), ("zbar1", zbar1), ("D", zbar1 - zbar), ("S", Sx), ("T", Tx), ("W", Wx), ("I", Ix)):
                        S_[f"{q}_{k}"].append(float(val))
            prev[b] = cur
            # reset ancestry: members -> self; non-member alive (born this step) -> genetic parent if member, else immigrant
            memset = set(cur["uid"].tolist())
            anc_new[b] = -1
            anc_new[b, mi] = uid[b, mi]
            nm = np.nonzero(al[b] & ~mem)[0]
            anc_new[b, nm] = np.where(np.isin(gpar[b, nm], list(memset)), gpar[b, nm], -1)
        st["anc"] = jnp.asarray(anc_new.astype(np.int32))
        if t == FORK_T:
            for b in range(B):
                np.savez_compressed(os.path.join(OUT, f"fork_{v}_s{SEEDS[b]}.npz"), **{k: np.asarray(x[b]) for k, x in st.items()})
        if t % 500 == 0:
            print(v, t, f"{time.time()-t0:.0f}s", flush=True)
            pickle.dump(dict(variant=v, t=t, series=series), open(os.path.join(OUT, f"partial_{v}.pkl"), "wb"))
    out = dict(variant=v, params=dict(BASE, **VARIANTS[v]), seeds=SEEDS, DT=DT, series=[{k: np.asarray(x) for k, x in s.items()} for s in series], wall_s=time.time() - t0)
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, time.time() - t0, "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["bank", "run", "one"]); ap.add_argument("--procs", type=int, default=7); ap.add_argument("--variant", default="full")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.kind == "bank":
        run_bank()
    elif a.kind == "one":
        print(worker(a.variant))
    else:
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for p, w, d in pool.imap_unordered(worker, list(VARIANTS)):
                print("DONE", os.path.basename(p), f"{w:.0f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
