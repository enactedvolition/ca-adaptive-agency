"""Exp 21 analysis (no-implant control, Exp 17 sw_pays t=3000 forks): pre-registered A21 = r_CL - r_OPEN, B21 = r_CL - r_FROZEN (paired 95% t CI, n = seeds),
and the specificity contrasts dA = A20 - A21, dB = B20 - B21 paired by seed (Exp 20 seed s started from the same sw_pays seed-s fork), Welch as robustness.
Per-seed metrics computed with analyze20's exact formulas. Writes figures/exp21/{tables21.md, report21.json, exp21_recovery.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp21"); os.makedirs(O, exist_ok=True)
import exp20_run as X
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean() if n else float("nan")
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n

def per_seed(d):
    A = d["arms"]; ref = A["ref"]; E0 = d["E0"]; w = slice(X.T // 2, None); o = {}
    nob = (A["ref"]["births"].sum(0) == 0) & (A["cl"]["births"].sum(0) == 0) & (A["open"]["births"].sum(0) == 0) & (A["frozen"]["births"].sum(0) == 0)
    hi = E0 > 0.8
    for arm in ("twin", "cl", "open", "frozen"):
        a = A[arm]; r = X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0); rE = X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0, offspring=False)
        o[arm] = dict(r=float(r.mean()), rE=float(rE.mean()), r_nobirth=float(rE[nob].mean()) if nob.any() else float("nan"), r_hi=float(r[hi].mean()) if hi.any() else float("nan"), r_lo=float(r[~hi].mean()) if (~hi).any() else float("nan"),
                      births=float(a["births"][w].sum(0).mean()), surv=float(a["alive"][-1].mean()), births_all=float(a["births"].sum(0).mean()))
    o["ref"] = dict(births=float(ref["births"][w].sum(0).mean()), surv=float(ref["alive"][-1].mean()), births_all=float(ref["births"].sum(0).mean()))
    o["n_nobirth"] = int(nob.sum()); o["n_hi"] = int(hi.sum()); o["K"] = d["K"]; o["clamp0_maxdiff_E"] = d["clamp0_maxdiff_E"]; o["clamp0_maxdiff_P"] = d["clamp0_maxdiff_P"]
    o["E0_mean"] = float(np.mean(E0)); o["pop0"] = d["pop0"]; o["ref_rep_rate"] = d.get("ref_rep_rate", float("nan"))
    return o

def load(dirname, pat):
    runs = {}
    for f in sorted(glob.glob(os.path.join(HERE, dirname, pat))):
        d = pickle.load(open(f, "rb")); runs[d["seed"]] = d
    return runs
r21, r20 = load("runs21", "run21_s*.pkl"), load("runs20", "run20_s*.pkl")
P21 = {s: per_seed(d) for s, d in r21.items()}; P20 = {s: per_seed(d) for s, d in r20.items()}
seeds = sorted(P21); rep = dict(seeds=seeds, n=len(seeds), per_seed21=P21)
ok = lambda P, s: P[s]["clamp0_maxdiff_E"] == 0 and P[s]["clamp0_maxdiff_P"] == 0
valid = all(ok(P21, s) for s in seeds); rep["clamp0_exact_all_seeds"] = bool(valid)
L = [f"# Exp 21 tables (no-implant control: Exp 20 assay on Exp 17 sw_pays t=3000 forks; sim13)\n\nn = {len(seeds)} seeds {seeds}; 40 targets/seed (random live, E0 >= 0.3), 50% E drop, T = {X.T}, window (T/2, T]\n",
     f"clamp0 (OPEN clamp, no drop) == ref exactly in all seeds: **{valid}** (max |dE| per seed: {[P21[s]['clamp0_maxdiff_E'] for s in seeds]}; max |d passed|: {[P21[s]['clamp0_maxdiff_P'] for s in seeds]})\n",
     "| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"]
def row(name, f, ss=None):
    ss = seeds if ss is None else ss; v = [f(s) for s in ss]; m, lo, hi, n = ci(v); rep[name] = dict(seeds=ss, per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v if not np.isnan(x))))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[name]
rA = row("PRIMARY 1: A21 = r_CL - r_OPEN", lambda s: P21[s]["cl"]["r"] - P21[s]["open"]["r"])
rB = row("PRIMARY 1: B21 = r_CL - r_FROZEN", lambda s: P21[s]["cl"]["r"] - P21[s]["frozen"]["r"])
for arm in ("cl", "open", "frozen", "twin"): row(f"r_{arm} (lineage energy, offspring-inclusive)", lambda s, a=arm: P21[s][a]["r"])
for arm in ("cl", "open", "frozen", "twin"): row(f"r_E_{arm} (own stored E only)", lambda s, a=arm: P21[s][a]["rE"])
row("secondary A_E = r_E_CL - r_E_OPEN", lambda s: P21[s]["cl"]["rE"] - P21[s]["open"]["rE"]); row("secondary B_E = r_E_CL - r_E_FROZEN", lambda s: P21[s]["cl"]["rE"] - P21[s]["frozen"]["rE"])
row("no-birth targets: A = r_E_CL - r_E_OPEN", lambda s: P21[s]["cl"]["r_nobirth"] - P21[s]["open"]["r_nobirth"]); row("no-birth targets: B = r_E_CL - r_E_FROZEN", lambda s: P21[s]["cl"]["r_nobirth"] - P21[s]["frozen"]["r_nobirth"])
row("E0>0.8 targets: A", lambda s: P21[s]["cl"]["r_hi"] - P21[s]["open"]["r_hi"]); row("E0>0.8 targets: B", lambda s: P21[s]["cl"]["r_hi"] - P21[s]["frozen"]["r_hi"])
row("E0<=0.8 targets: A", lambda s: P21[s]["cl"]["r_lo"] - P21[s]["open"]["r_lo"]); row("E0<=0.8 targets: B", lambda s: P21[s]["cl"]["r_lo"] - P21[s]["frozen"]["r_lo"])
for arm in ("ref", "cl", "open", "frozen"): row(f"births per target, whole run, {arm}", lambda s, a=arm: P21[s][a]["births_all"])
for arm in ("ref", "cl", "open", "frozen"): row(f"target survival at T, {arm}", lambda s, a=arm: P21[s][a]["surv"])
row("targets with no births in any arm (count)", lambda s: float(P21[s]["n_nobirth"])); row("targets with E0 > 0.8 (count)", lambda s: float(P21[s]["n_hi"]))
row("mean target E0", lambda s: P21[s]["E0_mean"]); row("ref targets' rep-action > 0 fraction while alive", lambda s: P21[s]["ref_rep_rate"])
p1 = bool(valid and rA["lo"] > 0 and rB["lo"] > 0); rep["primary1_pass"] = p1
L.append(f"\n**PRIMARY 1 (A21 and B21 CI both > 0, clamp0 exact): {'PASS' if p1 else 'FAIL'}** (A21 lo = {rA['lo']:+.3f}, B21 lo = {rB['lo']:+.3f})\n")
# ---- PRIMARY 2: specificity, paired by seed (same Exp 17 sw_pays seed-s fork lineage); seeds valid in both
pair = [s for s in seeds if s in P20 and ok(P20, s) and ok(P21, s)]; rep["paired_seeds"] = pair
L.append(f"## PRIMARY 2: Exp 20 (TRUE implant, t=1500) minus Exp 21 (no implant), paired by seed {pair} (n = {len(pair)})\n\n| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|")
row("Exp 20 A20 (recomputed)", lambda s: P20[s]["cl"]["r"] - P20[s]["open"]["r"], pair); row("Exp 20 B20 (recomputed)", lambda s: P20[s]["cl"]["r"] - P20[s]["frozen"]["r"], pair)
dA = row("PRIMARY 2: dA = A20 - A21", lambda s: (P20[s]["cl"]["r"] - P20[s]["open"]["r"]) - (P21[s]["cl"]["r"] - P21[s]["open"]["r"]), pair)
dB = row("PRIMARY 2: dB = B20 - B21", lambda s: (P20[s]["cl"]["r"] - P20[s]["frozen"]["r"]) - (P21[s]["cl"]["r"] - P21[s]["frozen"]["r"]), pair)
for nm, a1, a2 in (("dA no-birth own-E", "open", None), ("dB no-birth own-E", "frozen", None)):
    row(f"secondary {nm} (Exp20 - Exp21)", lambda s, a=a1: (P20[s]["cl"]["r_nobirth"] - P20[s][a]["r_nobirth"]) - (P21[s]["cl"]["r_nobirth"] - P21[s][a]["r_nobirth"]), pair)
row("secondary E0<=0.8 stratum dA", lambda s: (P20[s]["cl"]["r_lo"] - P20[s]["open"]["r_lo"]) - (P21[s]["cl"]["r_lo"] - P21[s]["open"]["r_lo"]), pair)
row("secondary E0<=0.8 stratum dB", lambda s: (P20[s]["cl"]["r_lo"] - P20[s]["frozen"]["r_lo"]) - (P21[s]["cl"]["r_lo"] - P21[s]["frozen"]["r_lo"]), pair)
for arm in ("cl", "open", "frozen"): row(f"context (post hoc): r_{arm}, Exp20 - Exp21", lambda s, a=arm: P20[s][a]["r"] - P21[s][a]["r"], pair)
row("context: twin r, Exp20 - Exp21", lambda s: P20[s]["twin"]["r"] - P21[s]["twin"]["r"], pair)
row("context: mean target E0, Exp20 - Exp21", lambda s: P20[s]["E0_mean"] - P21[s]["E0_mean"], pair)
def welch(a, b):
    a, b = np.asarray(a), np.asarray(b); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb); dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); m = a.mean() - b.mean(); h = stats.t.ppf(0.975, dfw) * se
    return float(m), float(m - h), float(m + h), float(dfw)
s20 = [s for s in sorted(P20) if ok(P20, s)]; s21 = [s for s in seeds if ok(P21, s)]
for nm, a in (("A", "open"), ("B", "frozen")):
    w_ = welch([P20[s]["cl"]["r"] - P20[s][a]["r"] for s in s20], [P21[s]["cl"]["r"] - P21[s][a]["r"] for s in s21]); rep[f"welch d{nm}"] = w_
    L.append(f"\n- robustness, unpaired Welch d{nm}: {w_[0]:+.3f} [{w_[1]:+.3f}, {w_[2]:+.3f}] (df {w_[3]:.1f})")
spec = (not p1) and dA["lo"] > 0 and dB["lo"] > 0; world = p1 and dA["lo"] <= 0 <= dA["hi"] and dB["lo"] <= 0 <= dB["hi"]
verdict = "SPECIFIC to the energy-aware (implanted) brains" if spec else ("PRODUCED BY THE WORLD/PROTOCOL" if world else "MIXED / NOT RESOLVED")
rep["verdict"] = verdict; L.append(f"\n**Pre-registered decision rule: {verdict}** (primary 1 {'PASS' if p1 else 'FAIL'}; dA lo {dA['lo']:+.3f}, dB lo {dB['lo']:+.3f})\n")
open(os.path.join(O, "tables21.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report21.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for j, (key, ttl) in enumerate((("r", "Exp 21 no implant: lineage-energy recovery r"), ("rE", "Exp 21 no implant: own-E recovery r_E"))):
    for i, arm in enumerate(("twin", "cl", "open", "frozen")):
        v = [P21[s][arm][key] for s in seeds]; ax[j].scatter(np.full(len(v), i) + np.linspace(-.1, .1, len(v)), v, s=14); ax[j].plot([i - .25, i + .25], [np.mean(v)] * 2, "k")
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(["twin (noise)", "CL", "OPEN", "FROZEN"]); ax[j].set_title(ttl, fontsize=9); ax[j].axhline(0, c="gray", lw=.5)
for i, (nm, a) in enumerate((("A", "open"), ("B", "frozen"))):
    for k, (P, lab, col) in enumerate(((P20, "Exp 20 implant", "C3"), (P21, "Exp 21 none", "C0"))):
        v = [P[s]["cl"]["r"] - P[s][a]["r"] for s in pair]; x = 2 * i + k
        ax[2].scatter(np.full(len(v), x) + np.linspace(-.1, .1, len(v)), v, s=14, c=col, label=lab if i == 0 else None); m, lo, hi, _ = ci(v); ax[2].errorbar(x + .25, m, [[m - lo], [hi - m]], fmt="ks", ms=4)
    for s in pair: ax[2].plot([2 * i, 2 * i + 1], [P20[s]["cl"]["r"] - P20[s][a]["r"], P21[s]["cl"]["r"] - P21[s][a]["r"]], c="gray", lw=.4)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["A20", "A21", "B20", "B21"]); ax[2].axhline(0, c="gray", lw=.5); ax[2].legend(fontsize=8); ax[2].set_title("A, B by seed (lines = paired seed), 95% t CI", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(O, "exp21_recovery.png"), dpi=110); print("\n".join(L))
