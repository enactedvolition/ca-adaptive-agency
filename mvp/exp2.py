"""Experiment runner for sim2: batched, multi-process sweeps + metric-grade main runs.

  python exp2.py sweep  --out runs2/sweep   # grid A (cost x mutation) + grid B (cost x env variability)
  python exp2.py main   --out runs2/main    # N=2048 showcase runs, 6 variants x seeds, dense recording
Each job batch = up to B sims sharing a static Cfg (params are traced, so one compile per Cfg)."""
from __future__ import annotations
import argparse, itertools, json, os, pickle, sys, time, hashlib, platform
from dataclasses import asdict, replace
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COSTS = [0.002, 0.004, 0.007, 0.011]
MUTS = [0.01, 0.03, 0.1, 0.3]
ENVS = [0.0, 0.15, 0.35, 0.7]
MUT_B = 0.05
ENV_A = 0.35
if os.environ.get("CA_GRID") == "big":          # GPU-scale grid (see modal_sweep.py)
    COSTS = [0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.011, 0.015]
    MUTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
    ENVS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4]

def sweep_jobs(seeds):
    J = []
    for c, m, s in itertools.product(COSTS, MUTS, seeds):
        J.append(dict(grid="A", variant="full", c_basal=c, mut_sigma=m, env_var=ENV_A, seed=s))
        J.append(dict(grid="A", variant="shuffled", c_basal=c, mut_sigma=m, env_var=ENV_A, seed=s, shuffle=1.0))
    for c, s in itertools.product(COSTS, seeds):
        J.append(dict(grid="A", variant="no_mut", c_basal=c, mut_sigma=0.0, env_var=ENV_A, seed=s))
    for m, s in itertools.product(MUTS, seeds):
        J.append(dict(grid="A", variant="no_cost", c_basal=0.004, mut_sigma=m, env_var=ENV_A, seed=s, cost_mul=0.0))
    for c, e, s in itertools.product(COSTS, ENVS, seeds):
        J.append(dict(grid="B", variant="full", c_basal=c, mut_sigma=MUT_B, env_var=e, seed=s))
        J.append(dict(grid="B", variant="no_build", c_basal=c, mut_sigma=MUT_B, env_var=e, seed=s, build_eff=0.0))
        J.append(dict(grid="B", variant="shuffled", c_basal=c, mut_sigma=MUT_B, env_var=e, seed=s, shuffle=1.0))
    return J

MAIN_VARIANTS = {"full": {}, "shuffled": dict(shuffle=1.0), "no_cost": dict(cost_mul=0.0), "no_mut": dict(mut_sigma=0.0),
                 "static_env": dict(feedback=0.0), "no_build": dict(build_eff=0.0)}

def main_jobs(seeds):
    return [dict(dict(grid="main", variant=v, c_basal=0.004, mut_sigma=MUT_B, env_var=ENV_A, seed=s), **kw)
            for v, kw in MAIN_VARIANTS.items() for s in seeds]

PKEYS = ("c_basal", "mut_sigma", "env_var", "cost_mul", "shuffle", "feedback", "build_eff")

def worker(args):
    jobs, cfg_kw, steps, dense_every, out_path = args
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)
    import jax, jax.numpy as jnp
    import sim2, metrics2 as M2
    cfg = sim2.Cfg(**cfg_kw)
    binit, bchunk, bcolour, consts = sim2.batched(cfg)
    B = len(jobs)
    plist = [sim2.params(**{k: j[k] for k in PKEYS if k in j}) for j in jobs]
    prm = sim2.stack_params(plist)
    keys = jnp.stack([jax.random.PRNGKey(1000 * j["seed"] + 7) for j in jobs])
    st = binit(keys, prm)
    H, G, cs = cfg.H, cfg.G, cfg.L / cfg.G
    hh = consts["hh_off"]
    t0 = time.time()
    CH = 10
    ts = {j: dict(step=[], pop=[], lam_hist=[], lam_med=[], surp=[], simpson=[], lam_by_root=[]) for j in range(B)}
    dense = {j: [] for j in range(B)}
    window = {j: [] for j in range(B)}
    n_chunks = steps // CH
    win_start = steps - 10 * CH
    def grab(st, keys_):
        return {k: np.asarray(st[k]) for k in keys_}
    for i in range(n_chunks):
        st = bchunk(st, prm, CH)
        t = (i + 1) * CH
        light = (t % 50 == 0) or (dense_every and t % dense_every == 0)
        heavy = t > win_start
        if not (light or heavy):
            continue
        g = grab(st, ["alive", "lam_sum", "lam_n", "surp", "root", "birth", "uid", "pos", "E", "n_off", "res", "spd", "own",
                      "gain_sum", "prate", "h", "xprev", "act", "mse", "xv", "intake", "W"] + (["Rf", "Wl", "Fvar", "F"] if (dense_every and t % dense_every == 0) or t == steps else []))
        for b in range(B):
            a = g["alive"][b]; ln = g["lam_n"][b]
            lam = np.where(ln >= 20, g["lam_sum"][b] / np.maximum(ln, 1), np.nan)
            age = t - g["birth"][b]
            if t % 50 == 0:
                ts[b]["step"].append(t); ts[b]["pop"].append(int(a.sum()))
                la = lam[a]
                ts[b]["lam_hist"].append(np.histogram(la[np.isfinite(la)], M2.LAM_BINS)[0])
                ts[b]["lam_med"].append(float(np.nanmedian(la)) if np.isfinite(la).any() else np.nan)
                sa = g["surp"][b][a & (age >= 50)]
                ts[b]["surp"].append(float(sa.mean()) if len(sa) else np.nan)
                ts[b]["simpson"].append(M2.simpson(g["root"][b][a]))
            if dense_every and t % dense_every == 0:
                cell = np.clip((g["pos"][b][a] / cs).astype(int), 0, G - 1)
                cols = np.asarray(bcolour(jnp.asarray(g["W"][b][a][None])))[0] if False else None
                dense[b].append(dict(t=t, pos=g["pos"][b][a].astype(np.float16), lam=lam[a].astype(np.float16),
                                     root=g["root"][b][a], uid=g["uid"][b][a], surp=g["surp"][b][a].astype(np.float16),
                                     age=age[a], E=g["E"][b][a].astype(np.float16), n_off=g["n_off"][b][a],
                                     Rf=g["Rf"][b].astype(np.float16), Wl=g["Wl"][b].astype(np.float16), Fvar=g["Fvar"][b].astype(np.float16)))
            if heavy:
                cell = np.clip((g["pos"][b][a] / cs).astype(int), 0, G - 1)
                Wl_here = np.asarray(st["Wl"][b])[cell[:, 0], cell[:, 1]] if True else None
                Fv_here = np.asarray(st["Fvar"][b])[cell[:, 0], cell[:, 1]]
                xv = g["xv"][b][a]
                window[b].append(dict(t=t, uid=g["uid"][b][a], root=g["root"][b][a], h=g["h"][b][a], x=g["xprev"][b][a],
                                      act=g["act"][b][a], age=age[a].astype(float), n_off=g["n_off"][b][a].astype(float),
                                      prate=g["prate"][b][a], lam=lam[a], gain=g["gain_sum"][b][a] / np.maximum(ln[a], 1),
                                      surp=g["surp"][b][a], mse=g["mse"][b][a], xvar=xv.mean(1),
                                      sent=0.5 * np.log(2 * np.pi * np.e * np.maximum(xv, 1e-6)).sum(1),
                                      res=g["res"][b][a], spd=g["spd"][b][a], own=g["own"][b][a], intake=g["intake"][b][a],
                                      E=g["E"][b][a], cx=cell[:, 0], cy=cell[:, 1], W_here=Wl_here, Fvar_here=Fv_here,
                                      build_eff=float(jobs[b].get("build_eff", 1.0)),
                                      Whh=g["W"][b][a][:, hh:hh + H * H].reshape(-1, H, H)))
    wall = time.time() - t0
    final_fields = {b: dict(Rf=np.asarray(st["Rf"][b]), Wl=np.asarray(st["Wl"][b]), Fvar=np.asarray(st["Fvar"][b])) for b in range(B)}
    ov = np.asarray(st["overflow"]); resc = np.asarray(st["rescued"])
    results = []
    v_term = cfg.a_max / (1 - cfg.damp)
    for b, j in enumerate(jobs):
        W = window[b]; last = W[-1]
        lam_end = last["lam"][np.isfinite(last["lam"]) & (last["age"] >= 60)]
        lam0_hist = ts[b]["lam_hist"][3]           # step 200: mostly founders
        centers = 0.5 * (M2.LAM_BINS[1:] + M2.LAM_BINS[:-1])
        lam0 = np.repeat(centers, lam0_hist)
        rho = np.max(np.abs(np.linalg.eigvals(last["Whh"])), axis=1) if len(last["uid"]) else np.array([])
        hfit, n_h = M2.heritability(last)
        info = M2.info_metrics(W, M2_EXT)
        reg = M2.regimes(last, cfg, j["env_var"], consts["Amap"], v_term)
        reg_frac = {r: float(np.mean(reg == i)) if len(reg) else np.nan for i, r in enumerate(M2.REGIMES)}
        rate = last["n_off"] / np.maximum(last["age"], 1)
        per_reg = {}
        for i, r in enumerate(M2.REGIMES):
            m = reg == i
            per_reg[r] = dict(n=int(m.sum()), surp=float(np.mean(last["surp"][m])) if m.any() else np.nan,
                              intake=float(np.mean(last["intake"][m])) if m.any() else np.nan,
                              rate=float(np.mean(rate[m])) if m.any() else np.nan, E=float(np.mean(last["E"][m])) if m.any() else np.nan,
                              xvar=float(np.mean(last["xvar"][m])) if m.any() else np.nan)
        pops = np.array(ts[b]["pop"])
        res = dict(job=j, wall_batch=wall, overflow=int(ov[b]), rescued=int(resc[b]), nan_h=int(np.isnan(last["h"]).sum()),
                   pop_end=int(len(last["uid"])), pop_mean_late=float(pops[len(pops) // 2:].mean()),
                   survived=bool(pops[-1] >= cfg.min_pop_frac * cfg.N and resc[b] == 0),
                   simpson_end=M2.simpson(last["root"]), herit=hfit, n_herit=n_h,
                   rho_med=float(np.median(rho)) if len(rho) else np.nan, gain_med=float(np.nanmedian(last["gain"])) if len(last["gain"]) else np.nan,
                   lam0=M2.lam_stats(lam0), lam_end=M2.lam_stats(lam_end), **info, regime_frac=reg_frac, per_regime=per_reg,
                   surp_med=float(np.median(last["surp"][last["age"] >= 50])) if (last["age"] >= 50).any() else np.nan,
                   xvar_med=float(np.median(last["xvar"])) if len(last["xvar"]) else np.nan,
                   ts={k: np.asarray(v) for k, v in ts[b].items() if k != "lam_by_root"})
        if dense_every:
            res["dense"] = dense[b]
            res["final"] = dict(lam=last["lam"], rate=rate, age=last["age"], surp=last["surp"], intake=last["intake"],
                                reg=reg, root=last["root"], rho=rho, gain=last["gain"], E=last["E"], xvar=last["xvar"])
            res["fields"] = final_fields[b]
        results.append(res)
    pickle.dump(dict(cfg=asdict(cfg), steps=steps, results=results), open(out_path, "wb"))
    return out_path, wall, B

M2_EXT = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]

def run(kind, out, seeds, steps, B, procs):
    os.makedirs(out, exist_ok=True)
    if kind == "sweep":
        jobs = sweep_jobs(seeds); cfg_kw = {}; dense_every = 0
    else:
        jobs = main_jobs(seeds); cfg_kw = dict(N=2048, L=1.0, G=64); dense_every = 20
    batches = [jobs[i:i + B] for i in range(0, len(jobs), B)]
    args = [(bt, cfg_kw, steps, dense_every, os.path.join(out, f"batch_{i:03d}.pkl")) for i, bt in enumerate(batches)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        for path, wall, nb in pool.imap_unordered(worker, args):
            print(f"{os.path.basename(path)}: {nb} sims, {wall:.0f}s  (elapsed {time.time()-t0:.0f}s)", flush=True)
    meta = dict(kind=kind, n_jobs=len(jobs), seeds=seeds, steps=steps, batch=B, procs=procs, wall_s=time.time() - t0,
                cpu_s_est=None, cfg_kw=cfg_kw, platform=platform.platform(), nproc=os.cpu_count(),
                code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in ("sim2.py", "metrics2.py", "exp2.py")})
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sweep", "main"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--procs", type=int, default=7)
    a = ap.parse_args()
    run(a.kind, a.out, list(range(a.seeds)), a.steps, a.batch, a.procs)
