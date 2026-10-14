"""Experiment 3a: bitwise replay of the saved main runs (runs2/main) with per-step identity/parent logging,
for an exact Price decomposition of mean lambda and Lande-Arnold selection gradients.

The replay uses the *unmodified* sim2 step, jitted exactly as in exp2 (same batch composition = same
variant x seeds 0..2, same PRNG keys), advanced one step per call (verified bitwise identical to the
saved trajectories: same uids and lambda at every saved 20-step snapshot). All extra logging is done
host-side from consecutive states; nothing feeds back into the dynamics.

  python exp3_price.py bank                 # sample the standardized input bank once (from full seed 0, steps 1001-1040)
  python exp3_price.py replay --procs 6     # 6 batches (variant x 3 seeds), checkpoint = one file per batch
"""
from __future__ import annotations
import argparse, os, pickle, sys, time, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("P3_OUT", os.path.join(HERE, "runs3"))
DT = 10                 # Price interval (steps)
FORK_T = 3000           # full-state snapshot for perturbation forks
STEPS = int(os.environ.get("P3_STEPS", 4000))
BANK_BURN, BANK_T = 10, 40

def setup():
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)

def load_batch_jobs(variant):
    import exp2
    jobs = [j for j in exp2.main_jobs([0, 1, 2]) if j["variant"] == variant]
    return jobs

def make_sim(jobs):
    import jax, jax.numpy as jnp, sim2, exp2
    cfg = sim2.Cfg(N=2048, L=1.0, G=64)
    binit, bchunk, bcol, consts = sim2.batched(cfg)
    prm = sim2.stack_params([sim2.params(**{k: j[k] for k in exp2.PKEYS if k in j}) for j in jobs])
    keys = jnp.stack([jax.random.PRNGKey(1000 * j["seed"] + 7) for j in jobs])
    return cfg, binit(keys, prm), bchunk, prm, consts

def bank_lambda_fn(cfg, consts, bank):
    """lambda of each brain driven by the shared bank, starting from a given hidden state (or zeros)."""
    import jax, jax.numpy as jnp, sim2
    lay, P, _ = sim2.layout(cfg.H); H = cfg.H
    bank = jnp.asarray(bank, jnp.float32)                    # (K, T, 15)
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
    f = jax.jit(jax.vmap(jax.vmap(one)))                     # (B, N, P), (B, N, H) -> (B, N)
    return f

def run_bank():
    setup()
    import jax, jax.numpy as jnp
    jobs = load_batch_jobs("full")
    cfg, st, bchunk, prm, consts = make_sim(jobs)
    for _ in range(100):
        st = bchunk(st, prm, 10)
    X, U = [], []
    for _ in range(BANK_T):
        st = bchunk(st, prm, 1)
        X.append(np.asarray(st["xprev"][0])); U.append(np.asarray(st["uid"][0]) * np.asarray(st["alive"][0]))
    X = np.stack(X); U = np.stack(U)
    ok = np.nonzero((U == U[0]).all(0) & (U[0] > 0))[0]
    pick = np.random.default_rng(0).choice(ok, 4, replace=False)
    bank = X[:, pick].transpose(1, 0, 2)                      # (4, 40, 15)
    np.savez(os.path.join(OUT, "input_bank.npz"), bank=bank, slots=pick, uids=U[0][pick], t0=1001, source="runs2/main full seed 0 (bitwise replay)")
    print("bank", bank.shape, "uids", U[0][pick])

def worker(variant):
    setup()
    import jax, jax.numpy as jnp, sim2
    path = os.path.join(OUT, f"price_{variant}.pkl")
    if os.path.exists(path):
        return path, 0.0, "cached"
    jobs = load_batch_jobs(variant)
    B = len(jobs)
    cfg, st, bchunk, prm, consts = make_sim(jobs)
    N, H, G = cfg.N, cfg.H, cfg.G; cs = cfg.L / cfg.G; hh = consts["hh_off"]
    pr = {k: np.asarray(v) for k, v in prm.items()}
    bank = np.load(os.path.join(OUT, "input_bank.npz"))["bank"]
    lamb = bank_lambda_fn(cfg, consts, bank)
    saved = [pickle.load(open(os.path.join(HERE, "runs2/main", f"batch_{i:03d}.pkl"), "rb")) for i in range(6)]
    sres = [r for d in saved for r in d["results"] if r["job"]["variant"] == variant]
    sres = sorted(sres, key=lambda r: r["job"]["seed"])
    t0 = time.time()
    g = lambda k: np.asarray(st[k])
    Wloc = g("W").copy()
    z = lambda: np.zeros((B, N))
    l0s, l0n, c_int, c_cost, c_act, c_eat, c_build = z(), z(), z(), z(), z(), z(), z()
    anc = np.where(g("alive"), g("uid"), -1).astype(np.int64)
    gpar = np.full((B, N), -1, np.int64)
    MAXU = 200000
    tab = {k: np.full((B, MAXU), np.nan) for k in ("birth", "last", "za", "lam10", "gain", "zb_last", "zb_sum", "zb_n", "lamb0", "rho", "wnorm",
                                                    "c_int", "c_cost", "c_act", "c_eat", "c_build", "n_off", "root", "gpar", "age")}
    alive_end = np.zeros((B, MAXU), bool)
    series = [dict(t=[], n=[], n1=[], **{f"{q}_{c}": [] for q in ("a", "b") for c in ("zbar", "zbar1", "D", "S", "T", "W", "I", "Wy", "Wm")},
                   n_surv=[], n_off=[], n_imm=[], n_orphan=[]) for _ in range(B)]
    prev = [None] * B
    diag = dict(verify_mismatch=[], shuffle_ambig=0, shuffle_births=0, donor_missing=0)
    old = {k: g(k) for k in ("alive", "uid", "h", "v_tan", "pos", "intake", "Pf", "root")}
    for t in range(1, STEPS + 1):
        st = bchunk(st, prm, 1)
        new = {k: g(k) for k in ("alive", "uid", "parent", "root", "h", "act", "intake")}
        for b in range(B):
            ao, an = old["alive"][b], new["alive"][b]
            surv = ao & an & (new["uid"][b] == old["uid"][b])
            s_idx = np.nonzero(surv)[0]
            Whh = Wloc[b, s_idx, hh:hh + H * H].reshape(-1, H, H)
            jv = (1 - new["h"][b][s_idx] ** 2) * np.einsum("nk,nkh->nh", old["v_tan"][b][s_idx], Whh)
            l0s[b, s_idx] += np.log(np.linalg.norm(jv, axis=1) + 1e-30); l0n[b, s_idx] += 1
            a = new["act"][b][s_idx]
            cell = np.clip((old["pos"][b][s_idx] / cs).astype(int), 0, G - 1)
            Ph = old["Pf"][b][cell[:, 0], cell[:, 1]]
            cost = (pr["c_basal"][b] + cfg.c_move * (a[:, :2] ** 2).sum(1) + cfg.c_eat * (a[:, 2] + 1) / 2 + cfg.c_sig * np.abs(a[:, 5])
                    + cfg.c_bond * np.maximum(a[:, 6], 0) + cfg.c_att * np.maximum(-a[:, 4], 0) + cfg.c_build * (a[:, 7] + 1) / 2
                    + pr["feedback"][b] * cfg.c_tox * Ph ** 2) * pr["cost_mul"][b]
            c_cost[b, s_idx] += cost
            c_int[b, s_idx] += (new["intake"][b][s_idx] - 0.97 * old["intake"][b][s_idx]) / 0.03
            c_act[b, s_idx] += (a[:, :2] ** 2).sum(1); c_eat[b, s_idx] += (a[:, 2] + 1) / 2; c_build[b, s_idx] += (a[:, 7] + 1) / 2
            nb = np.nonzero(an & ~surv)[0]
            if len(nb):
                Wn = np.asarray(st["W"][b][jnp.asarray(nb)])
                slot_of = {int(u): i for i, u in enumerate(old["uid"][b]) if surv[i]}
                for jx, sl in enumerate(nb):
                    p = int(new["parent"][b][sl])
                    if p < 0:
                        gpar[b, sl] = -1; anc[b, sl] = -1
                    else:
                        if pr["shuffle"][b] > 0.5:
                            cand = np.nonzero(surv & (old["root"][b] == new["root"][b][sl]))[0]
                            if len(cand) == 0: cand = np.nonzero(surv)[0]
                            d = np.linalg.norm(Wloc[b, cand] - Wn[jx], axis=1); o = np.argsort(d)
                            ds = int(cand[o[0]]); diag["shuffle_births"] += 1
                            if len(o) > 1 and d[o[1]] < 1.2 * d[o[0]]:
                                diag["shuffle_ambig"] += 1
                        else:
                            ds = slot_of.get(p, -1)
                        if ds < 0:
                            diag["donor_missing"] += 1; gpar[b, sl] = -1; anc[b, sl] = -1
                        else:
                            gpar[b, sl] = old["uid"][b][ds]; anc[b, sl] = anc[b, ds]
                    Wloc[b, sl] = Wn[jx]
                    for arr in (l0s, l0n, c_int, c_cost, c_act, c_eat, c_build):
                        arr[b, sl] = 0.0
        old = {k: (new[k] if k in new else g(k)) for k in ("alive", "uid", "h", "intake", "root")}
        old.update(v_tan=g("v_tan"), pos=g("pos"), Pf=g("Pf"))
        if t % DT == 0:
            zb_all = np.asarray(lamb(st["W"], st["h"]))
            zb0_all = None
            extra = {k: g(k) for k in ("lam_sum", "lam_n", "gain_sum", "n_off", "birth", "root")}
            for b in range(B):
                al = new["alive"][b]; uid = new["uid"][b]
                mem = al & (l0n[b] >= 1)
                mi = np.nonzero(mem)[0]
                cur = dict(uid=uid[mi].astype(np.int64), za=l0s[b, mi] / l0n[b, mi], zb=zb_all[b, mi].astype(np.float64), anc=anc[b, mi],
                           age=t - extra["birth"][b][mi])
                # per-uid table
                ai = np.nonzero(al)[0]; u = uid[ai]
                newu = u[np.isnan(tab["birth"][b, u])]
                if len(newu):
                    sl = ai[np.isnan(tab["birth"][b, u])]
                    if zb0_all is None:
                        zb0_all = np.asarray(lamb(st["W"], jnp.zeros_like(st["h"])))
                    Wh = Wloc[b, sl, hh:hh + H * H].reshape(-1, H, H)
                    tab["lamb0"][b, newu] = zb0_all[b, sl]
                    tab["rho"][b, newu] = np.abs(np.linalg.eigvals(Wh)).max(1)
                    tab["wnorm"][b, newu] = np.linalg.norm(Wloc[b, sl], axis=1)
                    tab["birth"][b, newu] = extra["birth"][b][sl]; tab["gpar"][b, newu] = gpar[b, sl]; tab["root"][b, newu] = extra["root"][b][sl]
                ln = extra["lam_n"][b][ai]
                tab["last"][b, u] = t; tab["age"][b, u] = t - extra["birth"][b][ai]
                tab["za"][b, u] = np.where(l0n[b, ai] >= 1, l0s[b, ai] / np.maximum(l0n[b, ai], 1), np.nan)
                tab["lam10"][b, u] = np.where(ln >= 20, extra["lam_sum"][b][ai] / np.maximum(ln, 1), np.nan)
                tab["gain"][b, u] = np.where(ln >= 1, extra["gain_sum"][b][ai] / np.maximum(ln, 1), np.nan)
                tab["zb_last"][b, u] = zb_all[b, ai]
                tab["zb_sum"][b, u] = np.nan_to_num(tab["zb_sum"][b, u]) + zb_all[b, ai]; tab["zb_n"][b, u] = np.nan_to_num(tab["zb_n"][b, u]) + 1
                for k, arr in (("c_int", c_int), ("c_cost", c_cost), ("c_act", c_act), ("c_eat", c_eat), ("c_build", c_build)):
                    tab[k][b, u] = arr[b, ai]
                tab["n_off"][b, u] = extra["n_off"][b][ai]
                if t == STEPS:
                    alive_end[b, u] = True
                # Price over (t-DT, t]
                P_ = prev[b]
                if P_ is not None:
                    S_ = series[b]; idx = {int(x): i for i, x in enumerate(P_["uid"])}
                    n, n1 = len(P_["uid"]), len(cur["uid"])
                    pi = np.array([idx.get(int(x), -1) for x in cur["anc"]])
                    is_s = (cur["anc"] == cur["uid"]) & (pi >= 0)
                    is_o = (~is_s) & (pi >= 0)
                    is_i = pi < 0
                    S_["n_orphan"].append(int(((cur["anc"] >= 0) & (pi < 0)).sum()))
                    w = np.zeros(n); np.add.at(w, pi[~is_i], 1.0)
                    S_["t"].append(t); S_["n"].append(n); S_["n1"].append(n1)
                    S_["n_surv"].append(int(is_s.sum())); S_["n_off"].append(int(is_o.sum())); S_["n_imm"].append(int(is_i.sum()))
                    for q in ("a", "b"):
                        zp = P_["z" + q]; zc = cur["z" + q]; zbar = zp.mean(); zbar1 = zc.mean()
                        Sx = (np.sum(w * zp) - w.mean() * n * zbar) / n1
                        Tx = np.sum(zc[is_o] - zp[pi[is_o]]) / n1
                        dlt = zc[is_s] - zp[pi[is_s]]
                        Wx = np.sum(dlt) / n1
                        young = cur["age"][is_s] < 50
                        Ix = np.sum(zc[is_i] - zbar) / n1
                        for k, v in (("zbar", zbar), ("zbar1", zbar1), ("D", zbar1 - zbar), ("S", Sx), ("T", Tx), ("W", Wx), ("I", Ix),
                                     ("Wy", np.sum(dlt[young]) / n1), ("Wm", np.sum(dlt[~young]) / n1)):
                            S_[f"{q}_{k}"].append(float(v))
                prev[b] = cur
                anc[b, mi] = uid[mi]
                nm = np.nonzero(al & ~mem)[0]                 # newborns of the last step -> attribute to genetic parent
                gp = gpar[b, nm]
                memset = set(uid[mi].tolist())
                anc[b, nm] = np.where(np.isin(gp, list(memset)), gp, -1)
            # verify bitwise replay against saved dense snapshots
            if t % 20 == 0:
                ls, lnn = extra["lam_sum"], extra["lam_n"]
                for b in range(B):
                    D = sres[b]["dense"][t // 20 - 1]; al = new["alive"][b]
                    lam = np.where(lnn[b] >= 20, ls[b] / np.maximum(lnn[b], 1), np.nan)[al].astype(np.float16)
                    if not (np.array_equal(new["uid"][b][al], D["uid"]) and np.array_equal(lam, D["lam"], equal_nan=True)):
                        diag["verify_mismatch"].append((b, t))
        if t == FORK_T:
            for b in range(B):
                np.savez_compressed(os.path.join(OUT, f"fork_{variant}_s{jobs[b]['seed']}.npz"),
                                    **{k: np.asarray(v[b]) for k, v in st.items()}, gpar=gpar[b], l0s=l0s[b], l0n=l0n[b])
        if t % 500 == 0:
            print(variant, t, f"{time.time()-t0:.0f}s", "mismatch", len(diag["verify_mismatch"]), flush=True)
    maxu = int(np.nanmax(tab["birth"] * 0 + np.arange(MAXU))) + 1 if False else MAXU
    used = int(np.max(np.where(~np.isnan(tab["birth"]), np.arange(MAXU)[None], -1))) + 1
    tab = {k: v[:, :used] for k, v in tab.items()}
    out = dict(variant=variant, jobs=jobs, DT=DT, series=[{k: np.asarray(v) for k, v in s.items()} for s in series], tab=tab,
               alive_end=alive_end[:, :used], diag=diag, wall_s=time.time() - t0, saved_ts_pop=[r["ts"]["pop"] for r in sres])
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, time.time() - t0, diag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["bank", "replay", "one"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--variant", default="full")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.kind == "bank":
        run_bank()
    elif a.kind == "one":
        print(worker(a.variant))
    else:
        V = ["full", "shuffled", "no_cost", "no_mut", "static_env", "no_build"]
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for path, wall, d in pool.imap_unordered(worker, V):
                print("DONE", os.path.basename(path), f"{wall:.0f}s", d if isinstance(d, str) else {k: (v if not isinstance(v, list) else len(v)) for k, v in d.items()}, f"elapsed {time.time()-t0:.0f}s", flush=True)
        json.dump(dict(wall_s=time.time() - t0, procs=a.procs), open(os.path.join(OUT, "price_meta.json"), "w"))
