"""Exp 20 analysis: pre-registered A = r_CL - r_OPEN, B = r_CL - r_FROZEN (offspring-inclusive lineage-energy recovery), paired 95% t CI, n = seeds.
Also secondary metrics and the post hoc threshold read. Writes figures/exp20/{tables20.md, report20.json, exp20_recovery.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs20"); O = os.path.join(HERE, "..", "figures", "exp20"); os.makedirs(O, exist_ok=True)
import exp20_run as X
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean() if n else float("nan")
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n

runs = {}
for f in sorted(glob.glob(os.path.join(R, "run20_s*.pkl"))):
    d = pickle.load(open(f, "rb")); runs[d["seed"]] = d
seeds = sorted(runs); rep = dict(seeds=seeds, n=len(seeds)); L = [f"# Exp 20 tables (perturbation recovery, Exp 19 TRUE population at t=1500; sim13)\n\nn = {len(seeds)} seeds {seeds}; 40 targets/seed, 50% E drop, T = {X.T}, window (T/2, T]\n"]
per = {}
for sd in seeds:
    d = runs[sd]; A = d["arms"]; ref = A["ref"]; E0 = d["E0"]; w = slice(X.T // 2, None); o = {}
    nob = (A["ref"]["births"].sum(0) == 0) & (A["cl"]["births"].sum(0) == 0) & (A["open"]["births"].sum(0) == 0) & (A["frozen"]["births"].sum(0) == 0)
    hi = E0 > 0.8
    for arm in ("twin", "cl", "open", "frozen"):
        a = A[arm]; r = X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0); rE = X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0, offspring=False)
        o[arm] = dict(r=float(r.mean()), rE=float(rE.mean()), r_nobirth=float(rE[nob].mean()) if nob.any() else float("nan"), r_hi=float(r[hi].mean()) if hi.any() else float("nan"), r_lo=float(r[~hi].mean()) if (~hi).any() else float("nan"),
                      births=float(a["births"][w].sum(0).mean()), surv=float(a["alive"][-1].mean()), births_all=float(a["births"].sum(0).mean()))
    o["ref"] = dict(births=float(ref["births"][w].sum(0).mean()), surv=float(ref["alive"][-1].mean()), births_all=float(ref["births"].sum(0).mean()))
    o["n_nobirth"] = int(nob.sum()); o["n_hi"] = int(hi.sum()); o["K"] = d["K"]; o["clamp0_maxdiff_E"] = d["clamp0_maxdiff_E"]; o["clamp0_maxdiff_P"] = d["clamp0_maxdiff_P"]; per[sd] = o
rep["per_seed"] = per
valid = all(per[s]["clamp0_maxdiff_E"] == 0 and per[s]["clamp0_maxdiff_P"] == 0 for s in seeds); rep["clamp0_exact_all_seeds"] = bool(valid)
L.append(f"clamp0 (OPEN clamp, no drop) == ref exactly in all seeds: **{valid}** (max |dE| per seed: {[per[s]['clamp0_maxdiff_E'] for s in seeds]})\n")
L.append("| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|")
def row(name, f):
    v = [f(s) for s in seeds]; m, lo, hi, n = ci(v); rep[name] = dict(per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v if not np.isnan(x))))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |")
row("PRIMARY A = r_CL - r_OPEN", lambda s: per[s]["cl"]["r"] - per[s]["open"]["r"])
row("PRIMARY B = r_CL - r_FROZEN", lambda s: per[s]["cl"]["r"] - per[s]["frozen"]["r"])
for arm in ("cl", "open", "frozen", "twin"): row(f"r_{arm} (lineage energy, offspring-inclusive)", lambda s, a=arm: per[s][a]["r"])
for arm in ("cl", "open", "frozen", "twin"): row(f"r_E_{arm} (own stored E only; Exp 6 style)", lambda s, a=arm: per[s][a]["rE"])
row("secondary A_E = r_E_CL - r_E_OPEN", lambda s: per[s]["cl"]["rE"] - per[s]["open"]["rE"]); row("secondary B_E = r_E_CL - r_E_FROZEN", lambda s: per[s]["cl"]["rE"] - per[s]["frozen"]["rE"])
row("no-birth targets: A = r_E_CL - r_E_OPEN", lambda s: per[s]["cl"]["r_nobirth"] - per[s]["open"]["r_nobirth"]); row("no-birth targets: B = r_E_CL - r_E_FROZEN", lambda s: per[s]["cl"]["r_nobirth"] - per[s]["frozen"]["r_nobirth"])
row("E0>0.8 targets: A", lambda s: per[s]["cl"]["r_hi"] - per[s]["open"]["r_hi"]); row("E0>0.8 targets: B", lambda s: per[s]["cl"]["r_hi"] - per[s]["frozen"]["r_hi"])
row("E0<=0.8 targets: A", lambda s: per[s]["cl"]["r_lo"] - per[s]["open"]["r_lo"]); row("E0<=0.8 targets: B", lambda s: per[s]["cl"]["r_lo"] - per[s]["frozen"]["r_lo"])
for arm in ("ref", "cl", "open", "frozen"): row(f"births per target, whole run, {arm}", lambda s, a=arm: per[s][a]["births_all"])
for arm in ("ref", "cl", "open", "frozen"): row(f"target survival at T, {arm}", lambda s, a=arm: per[s][a]["surv"])
row("targets with no births in any arm (count)", lambda s: float(per[s]["n_nobirth"]))
rA, rB = rep["PRIMARY A = r_CL - r_OPEN"], rep["PRIMARY B = r_CL - r_FROZEN"]
rep["primary_pass"] = bool(valid and rA["lo"] > 0 and rB["lo"] > 0); rep["A_excludes0_below"] = bool(rA["lo"] > 0); rep["B_excludes0_below"] = bool(rB["lo"] > 0)
L.append(f"\n**Pre-registered rule (A and B CI both > 0, clamp0 exact): {'PASS' if rep['primary_pass'] else 'FAIL'}** (A lo = {rA['lo']:+.3f}, B lo = {rB['lo']:+.3f})\n")
# ---- post hoc: implied rep threshold from m = 1 genomes
import exp19_run as X19
cfg, chunk_one, consts, prm, tmpl, off = X19.world(); L.append("## Post hoc: implied rep threshold (implant unit k*; ignores other inputs)\n\n| seed | n live m=1 | E_thr median [IQR] | gain W_ih[8,k*] median | frac m=1 with E_thr within 0.8 +-0.05 | live-E frac > 0.8 at t=0 | at t=1500 | frac of m=1 with E > own E_thr (t=1500) |\n|---|---|---|---|---|---|---|---|")
th = {}
for sd in seeds:
    z = np.load(os.path.join(R, f"fork20_s{sd}_t1500.npz")); W, al, E = z["W"], z["alive"], z["E"]; oi, (I, H) = off["W_ih"][0], off["W_ih"][1]; oo, (_, A) = off["W_ho"][0], off["W_ho"][1]
    wi8 = W[:, oi + 8 * H: oi + 9 * H]; wo3 = W[:, oo + 3: oo + H * A: A]; sc = wi8 * wo3; k = sc.argmax(1); m = (sc.max(1) > X19.MARK) & al; ix = np.arange(len(W))
    g = wi8[ix, k]; b = W[:, off["b_h"][0] + k] if False else W[ix, off["b_h"][0] + k]; ethr = np.expm1(-b / np.where(np.abs(g) > 1e-9, g, np.nan) + 0.7)
    z0 = np.load(os.path.join(HERE, "runs17", f"fork16_sw_pays_s{sd}_t3000.npz")); e0 = z0["E"][z0["alive"]]
    q = np.nanpercentile(ethr[m], [25, 50, 75]); th[sd] = dict(n=int(m.sum()), ethr_q=q.tolist(), gain_med=float(np.median(g[m])), near=float(np.mean(np.abs(ethr[m] - 0.8) < 0.05)), f0=float((e0 > 0.8).mean()), f1=float((E[al] > 0.8).mean()), own=float((E[m] > ethr[m]).mean()), Emean0=float(e0.mean()), Emean1=float(E[al].mean()))
    t_ = th[sd]; L.append(f"| {sd} | {t_['n']} | {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}] | {t_['gain_med']:.2f} | {t_['near']:.2f} | {t_['f0']:.3f} | {t_['f1']:.3f} | {t_['own']:.3f} |")
rep["threshold_readout"] = th
for nm, f in (("median E_thr", lambda s: th[s]["ethr_q"][1]), ("share live E > 0.8, t=1500 minus t=0", lambda s: th[s]["f1"] - th[s]["f0"]), ("mean live E, t=1500 minus t=0", lambda s: th[s]["Emean1"] - th[s]["Emean0"])):
    v = [f(s) for s in seeds]; m_, lo, hi, n = ci(v); rep["ph " + nm] = dict(mean=m_, lo=lo, hi=hi, per_seed=v); L.append(f"\n- {nm}: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {n})")
open(os.path.join(O, "tables20.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report20.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for j, (key, ttl) in enumerate((("r", "lineage-energy recovery r (offspring-inclusive)"), ("rE", "own-E recovery r_E"))):
    for i, arm in enumerate(("twin", "cl", "open", "frozen")):
        v = [per[s][arm][key] for s in seeds]; ax[j].scatter(np.full(len(v), i) + np.linspace(-.1, .1, len(v)), v, s=14); ax[j].plot([i - .25, i + .25], [np.mean(v)] * 2, "k")
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(["twin (noise)", "CL", "OPEN", "FROZEN"]); ax[j].set_title(ttl, fontsize=9); ax[j].axhline(0, c="gray", lw=.5)
plt.tight_layout(); plt.savefig(os.path.join(O, "exp20_recovery.png"), dpi=110); print("\n".join(L))
