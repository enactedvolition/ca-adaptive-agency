"""Rebuild step 1 phase A (pre-registered NIGHTLY_LOG 2026-11-28 05:03 PST, commit 10ed9ea): S0 sensor-only intervention on the
KNOWN-CONTROLLER ladder only. Constructed controllers = calibration, not evidence about evolved or real systems.
World: Exp 27 setup via assay_core (Exp 20 t=1500 forks, births off, brains blind and NOT intervened, haz in {0, 0.004}).
  python s0_ladder.py run --haz 0.004      (detached; one pickle per (system, haz, seed) in runs_s0/, resumes)
  python s0_ladder.py summary              (runs_s0/s0_ladder_summary.json + gate verdicts)"""
import os, sys, json, pickle, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs_s0"); sys.path.insert(0, HERE)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
T, K, E_MIN, THETA, B0 = 6, 200, 0.3, 0.40, 0.5
DS = [0.25, 0.5]; KS = [0, 1, 3, 5]; SEEDS = list(range(9)); GAINS = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 16]
NW = 33; NSPEC = 4 + NW

def spec(kind, g=0.0, b=B0, src=0, w=None):
    v = np.zeros(NSPEC, np.float32); v[:4] = (kind, g, b, src)
    if w is not None: v[4:] = w
    return v

def systems():
    S = {f"P{g}": spec(0, g) for g in GAINS}
    S["STEP"] = spec(1); S["CONST0.8"] = spec(0, 0.0, 0.8)
    for nm, f, src in (("ES", "es", 0), ("ES_ABL", "es", 2), ("ES_BLIND", "esblind", 2)):
        p = os.path.join(HERE, "runs27", f"es_{f}.npz")
        if os.path.exists(p): S[nm] = spec(2, src=src, w=np.load(p)["theta"].astype(np.float32))
    return S
IGNORES_INPUT = ["P0", "CONST0.8", "ES_ABL", "ES_BLIND"]

def make_policy(jnp, jax, G, cs):
    def pol(u, c, sp):      # Exp 27 controller (src 0 own/fed u, src 2 constant theta); src 1 not supported here
        kind, g, b, src = sp[0], sp[1], sp[2], sp[3]; w = sp[4:]
        u = jnp.where(src == 2, jnp.float32(THETA), u)
        p = jnp.clip(b + g * (THETA - u), 0.0, 1.0); st = (u < THETA).astype(jnp.float32)
        cell = jnp.clip((c["pos"] / cs).astype(jnp.int32), 0, G - 1); rh = c["Rf"][cell[:, 0], cell[:, 1]]
        x = jnp.stack([jnp.log1p(jnp.maximum(u, 0.0)) - 0.7, rh], 1)
        h = jnp.tanh(x @ w[:16].reshape(2, 8) + w[16:24]); m = jax.nn.sigmoid(h @ w[24:32] + w[32])
        return jnp.where(kind == 0, p, jnp.where(kind == 1, st, m))
    return pol

def run(haz):
    import jax, jax.numpy as jnp, exp19_run as X, assay_core as A
    os.makedirs(OUT, exist_ok=True)
    cfg, chunk_one, consts, prm, tmpl, off = X.world(); prm = dict(prm); prm["blind"] = jnp.float32(1.0)
    pol = make_policy(jnp, jax, cfg.G, cfg.L / cfg.G); S = systems()
    for seed in SEEDS:
        todo = [n for n in S if not os.path.exists(os.path.join(OUT, f"s0_{n}_h{haz:g}_s{seed}.pkl"))]
        if not todo: continue
        z = np.load(os.path.join(HERE, "runs20", f"fork20_s{seed}_t1500.npz"))
        base = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
        elig = np.nonzero(np.asarray(base["alive"]) & (np.asarray(base["E"]) >= E_MIN))[0]
        tg = np.sort(np.random.default_rng(31000 + seed).choice(elig, K, replace=False))
        ra = A.build(consts["step"], prm, cfg.N, tg, T, rep_off="all", haz=(None if haz == 0 else haz), policy=pol, brain_sense=False)
        hk = jax.random.PRNGKey(31500 + seed); E0 = np.asarray(base["E"])[tg].astype(np.float64)
        for n in todo:
            t0 = time.time(); sp = S[n]
            ref = ra(base, "ref", hkey=hk, pp=sp)
            res = dict(name=n, spec=sp, haz=haz, seed=seed, tg=tg, E0=E0, ref_eat=ref["eat"], ref_alive=ref["alive"], births=int(ref["births"].sum()))
            for d in DS:
                iv = ra(base, "sscale", hkey=hk, fix=1.0 - d, pp=sp)
                both = ref["alive"] & iv["alive"]
                res[f"d{d}"] = dict(eat=iv["eat"], alive=iv["alive"], u0=iv["u"][0], both=both,
                                    dEat={k: float(np.mean((iv["eat"][k] - ref["eat"][k])[both[k]])) for k in KS})
            res["secs"] = time.time() - t0
            p = os.path.join(OUT, f"s0_{n}_h{haz:g}_s{seed}.pkl"); pickle.dump(res, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
        print(f"seed {seed} haz {haz} done {time.strftime('%H:%M:%S')}", flush=True)

def summary():
    from scipy import stats
    S = systems(); out = dict(systems={}, gate={}); gate_ok = True
    for haz in (0.0, 0.004):
        rows = {}
        for n in S:
            ps = [os.path.join(OUT, f"s0_{n}_h{haz:g}_s{s}.pkl") for s in SEEDS]
            if not all(os.path.exists(p) for p in ps): continue
            R = [pickle.load(open(p, "rb")) for p in ps]; row = {}
            for d in DS:
                for k in KS:
                    v = np.array([r[f"d{d}"]["dEat"][k] for r in R]); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                    row[f"d{d}_k{k}"] = dict(mean=float(m), lo=float(m - h), hi=float(m + h))
                # exactness checks
                if n in IGNORES_INPUT:
                    row[f"d{d}_exact_zero"] = bool(all(np.array_equal(r[f"d{d}"]["eat"][r[f"d{d}"]["both"]], r["ref_eat"][r[f"d{d}"]["both"]]) for r in R))
                if S[n][0] == 0 and S[n][3] == 0:
                    g, b = float(S[n][1]), float(S[n][2])
                    err = max(float(np.max(np.abs((r[f"d{d}"]["eat"][0] - r["ref_eat"][0]) - (np.clip(b + g * (THETA - (1 - d) * r["E0"]), 0, 1) - np.clip(b + g * (THETA - r["E0"]), 0, 1))))) for r in R)
                    row[f"d{d}_k0_analytic_maxerr"] = err
            row["births"] = int(sum(r["births"] for r in R)); rows[n] = row
        out["systems"][f"haz{haz:g}"] = rows
        if not rows: continue
        g_i = all(rows[n].get(f"d{d}_exact_zero", False) for n in IGNORES_INPUT if n in rows for d in DS)
        g_ii = all(rows[f"P{g}"][f"d{d}_k0_analytic_maxerr"] <= 1e-6 for g in GAINS for d in DS)
        pm = [rows[f"P{g}"]["d0.5_k3"]["mean"] for g in GAINS]; rho = float(stats.spearmanr(GAINS, pm)[0])
        g_iii = rho >= 0.9 and all(rows[f"P{g}"]["d0.5_k3"]["lo"] > 0 for g in GAINS if g >= 0.05)
        g_iv = rows["STEP"]["d0.5_k3"]["lo"] > 0
        out["gate"][f"haz{haz:g}"] = dict(i_input_ignoring_exact_zero=g_i, ii_k0_analytic=g_ii, iii_monotone=g_iii, spearman=rho, primary_by_gain=pm, iv_step=g_iv)
        gate_ok &= g_i and g_ii and g_iii and g_iv
    out["gate"]["PASS"] = bool(gate_ok and len(out["gate"]) == 2)
    json.dump(out, open(os.path.join(OUT, "s0_ladder_summary.json"), "w"), indent=1); print(json.dumps(out["gate"], indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--haz", type=float, default=0.004); a = ap.parse_args()
    run(a.haz) if a.cmd == "run" else summary()
