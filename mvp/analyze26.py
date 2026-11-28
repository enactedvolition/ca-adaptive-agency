"""Exp 26 analysis (pre-registered in NIGHTLY_LOG 2026-11-28). PRIMARY: target survival at t=150 (K=200 targets/seed), paired over seeds, 95% t CI:
A_surv = CL-OPEN, B_surv = CL-FROZEN, A_GC = CL-GC; D = ref-CL (informativeness). (1) COND replication (runs26/cond26_s*.pkl); (2) TRUE/ctrl/SHAM (runs26/run26_*).
Secondaries: t=300, 80% drop. Writes figures/exp26/{tables26.md, report26.json, exp26_survival.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp26"); os.makedirs(O, exist_ok=True); R26 = os.path.join(HERE, "runs26")
def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
rep = {}; L = ["# Exp 26 tables: target-survival read-out (K = 200 targets/seed, paired over seeds, 95% t CI, df = n-1)\n"]
H = "| contrast | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"
def row(key, name, v):
    m, lo, hi, n = ci(v); rep[key] = dict(per_seed=[float(x) for x in v], mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v)))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[key]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[key]
sv = lambda al, t: float(np.asarray(al)[t - 1].mean())
fig_items = {}
# ---------- (1) COND
C = {}
for f in sorted(glob.glob(os.path.join(R26, "cond26_s*.pkl"))):
    d = pickle.load(open(f, "rb")); C[d["seed"]] = d
if C:
    ss = sorted(C); valid = all(C[s]["clamp0_maxdiff_E"] == 0.0 for s in ss)
    L.append(f"\n## (1) COND positive control, pre-registered replication (haz = 0.004, eat = 1 iff E < 0.40; fresh target rng 2626+seed and key 2700+seed; T = 150)\n\nseeds {ss}; targets/seed {[len(C[s]['tg']) for s in ss]}; **clamp0 == ref bit-for-bit (E): {valid}** (max |dE| {max(C[s]['clamp0_maxdiff_E'] for s in ss)}). Births off by construction (rep forced off for every agent).\n\n" + H)
    S = {s: {a: sv(C[s]["arms"][a]["alive"], 150) for a in C[s]["arms"]} for s in ss}
    for a in ("ref", "cl", "open", "frozen"): row(f"cond_surv_{a}", f"survival at t = 150, {a}", [S[s][a] for s in ss])
    rA = row("cond_Asurv", "**A_surv = surv_CL − surv_OPEN**", [S[s]["cl"] - S[s]["open"] for s in ss])
    rB = row("cond_Bsurv", "**B_surv = surv_CL − surv_FROZEN**", [S[s]["cl"] - S[s]["frozen"] for s in ss])
    row("cond_D", "D = surv_ref − surv_CL (cost of the drop)", [S[s]["ref"] - S[s]["cl"] for s in ss])
    row("cond_clamp0_surv", "surv_clamp0 − surv_ref (validity, must be 0)", [S[s]["clamp0"] - S[s]["ref"] for s in ss])
    rep["cond_verdict"] = "PASS" if (valid and rA["lo"] > 0 and rB["lo"] > 0) else "FAIL"
    L.append(f"\n**COND replication: {rep['cond_verdict']}** (needs clamp0 exact and both A_surv, B_surv lower bounds > 0). Exp 25 post hoc (K=200, old draws): A_surv +0.041, B_surv +0.087.")
    fig_items["COND"] = (rA, rB, None)
# ---------- (2) populations
P = {}
for f in sorted(glob.glob(os.path.join(R26, "run26_*_s*.pkl"))):
    d = pickle.load(open(f, "rb")); P.setdefault(d["pop"], {})[d["seed"]] = d
NM = dict(true="TRUE (implant)", ctrl="ctrl (no implant)", sham="SHAM (age-matched)")
for pop in ("true", "ctrl", "sham"):
    if pop not in P: continue
    ss = sorted(P[pop]); D = P[pop]
    valid = all(D[s]["clamp0_maxdiff_E"] == 0.0 and D[s]["clamp0_alive_equal"] for s in ss) and all(sum(D[s]["cfg"][c]["births_total"].values()) == 0 for s in ss for c in D[s]["cfg"])
    L.append(f"\n## (2) {NM[pop]}: seeds {ss}, K = {[D[s]['K'] for s in ss]}, eligible {[D[s]['n_eligible'] for s in ss]}\n\n**Validity: clamp0 == ref bit-for-bit (E, alive) and 0 target births in all arms/configs, all seeds: {valid}.** GC map: median per-target R² {np.nanmedian(np.concatenate([D[s]['fit']['r2'] for s in ss])):.2f}, pooled-fallback targets {int(sum(D[s]['fit']['pooled'].sum() for s in ss))}.")
    for cfgn, tl in (("d50", "50% drop"), ("d80", "80% drop (secondary)")):
        for t in (150, 300):
            S = {s: {a: sv(D[s]["cfg"][cfgn]["arms"][a]["alive"], t) for a in D[s]["cfg"][cfgn]["arms"]} for s in ss}
            tag = f"{pop}_{cfgn}_t{t}"; prim = (cfgn == "d50" and t == 150)
            L.append(f"\n### {NM[pop]}, {tl}, survival at t = {t}{'  (PRIMARY)' if prim else '  (secondary)'}\n\n" + H)
            for a in ("ref", "twin", "cl", "open", "frozen", "gc", "gc0"): row(f"{tag}_surv_{a}", f"survival, {a}", [S[s][a] for s in ss])
            rA = row(f"{tag}_Asurv", "**A_surv = CL − OPEN**", [S[s]["cl"] - S[s]["open"] for s in ss])
            rB = row(f"{tag}_Bsurv", "**B_surv = CL − FROZEN**", [S[s]["cl"] - S[s]["frozen"] for s in ss])
            rG = row(f"{tag}_AGC", "**A_GC = CL − GC (state-matched passive)**", [S[s]["cl"] - S[s]["gc"] for s in ss])
            row(f"{tag}_AGCb", "A_GC baseline-corrected: (CL − ref) − (GC − GC0)", [(S[s]["cl"] - S[s]["ref"]) - (S[s]["gc"] - S[s]["gc0"]) for s in ss])
            rD = row(f"{tag}_D", "D = ref − CL (cost of the drop)", [S[s]["ref"] - S[s]["cl"] for s in ss])
            row(f"{tag}_noise", "ref − twin (noise floor)", [S[s]["ref"] - S[s]["twin"] for s in ss])
            row(f"{tag}_clamp0", "clamp0 − ref (validity, must be 0)", [S[s]["clamp0"] - S[s]["ref"] for s in ss])
            det = valid and rA["lo"] > 0 and rB["lo"] > 0
            rep[tag + "_detected"] = bool(det); rep[tag + "_informative"] = bool(rD["lo"] > 0); rep[tag + "_agency"] = bool(det and rG["lo"] > 0)
            L.append(f"\n**{tag}: feedback detected = {det}; metric informative (D lower bound > 0) = {rD['lo'] > 0}; active beats state-matched passive (A_GC lower bound > 0, with detection) = {det and rG['lo'] > 0}.**")
            if prim: fig_items[NM[pop]] = (rA, rB, rG)
# TRUE vs ctrl descriptive
if "true" in P and "ctrl" in P:
    L.append("\n## Descriptive: TRUE − ctrl (unpaired Welch, primary config)\n\n| contrast | diff [95% Welch CI] |\n|---|---|")
    for k in ("Asurv", "Bsurv", "AGC"):
        a = np.array(rep[f"true_d50_t150_{k}"]["per_seed"]); b = np.array(rep[f"ctrl_d50_t150_{k}"]["per_seed"]); d = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); dfw = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)); h = stats.t.ppf(.975, dfw) * se
        L.append(f"| {k} | {d:+.3f} [{d - h:+.3f}, {d + h:+.3f}] |"); rep[f"true_minus_ctrl_{k}"] = [float(d), float(d - h), float(d + h)]
open(os.path.join(O, "tables26.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report26.json"), "w"), indent=1)
# ---------- figure
if fig_items:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8)); names = list(fig_items); w = 0.27
    for j, (lab, c) in enumerate((("A_surv  CL − OPEN", "#2a6f97"), ("B_surv  CL − FROZEN", "#e07a1f"), ("A_GC  CL − GC", "#6a994e"))):
        for i, nme in enumerate(names):
            r = fig_items[nme][j]
            if r is None: continue
            ax[0].bar(i + (j - 1) * w, r["mean"], w, color=c, label=lab if (i == 0 or (j == 2 and i == 1)) else None, yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], capsize=3)
            ax[0].scatter(np.full(len(r["per_seed"]), i + (j - 1) * w), r["per_seed"], s=8, color="k", alpha=.45, zorder=3)
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_ylabel("survival difference at t = 150 (K = 200/seed)")
    ax[0].set_title("Active − passive recovery, survival read-out\n(mean, 95% t CI, dots = seeds, n = 9)", fontsize=10); ax[0].legend(fontsize=8)
    pops = [p for p in ("true", "ctrl", "sham") if p in P]
    for i, p in enumerate(pops):
        for j, (a, c) in enumerate((("ref", "#888"), ("cl", "#2a6f97"), ("open", "#c1121f"), ("frozen", "#e07a1f"), ("gc", "#6a994e"))):
            v = rep[f"{p}_d50_t150_surv_{a}"]; ax[1].bar(i * 6 + j, v["mean"], 0.9, color=c, label=a if i == 0 else None, yerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], capsize=2)
    ax[1].set_xticks([i * 6 + 2 for i in range(len(pops))]); ax[1].set_xticklabels([NM[p] for p in pops], fontsize=8); ax[1].set_ylabel("target survival at t = 150")
    ax[1].set_title("Arm survival, 50% drop (no predation, births off)", fontsize=10); ax[1].legend(fontsize=8, ncol=5); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(O, "exp26_survival.png"), dpi=130)
print("\n".join(L[:12]))
