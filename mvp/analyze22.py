"""Exp 22 analysis (births off for targets, own-E recovery). PRIMARY (Exp 20 TRUE t=1500 forks): A = rE_CL - rE_OPEN, B = rE_CL - rE_FROZEN,
paired over seeds, 95% t CI; pass iff both CIs > 0, clamp0 exact and 0 target births in every arm, all seeds. Secondary: same in the no-implant
control (Exp 17 sw_pays t=3000) and dA, dB = TRUE - ctrl paired by seed. rE = exp20_run.recovery(..., offspring=False).
Writes figures/exp22/{tables22.md, report22.json, exp22_recovery.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp22"); os.makedirs(O, exist_ok=True)
import exp20_run as X
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean() if n else float("nan")
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n

def per_seed(d):
    A = d["arms"]; ref = A["ref"]; E0 = d["E0"]; o = {}
    for arm in ("twin", "cl", "open", "frozen"):
        a = A[arm]; rE = X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0, offspring=False)
        o[arm] = dict(rE=float(rE.mean()), surv=float(a["alive"][-1].mean()), Eend=float(a["E"][-1].mean()))
    o["ref"] = dict(surv=float(ref["alive"][-1].mean()), Eend=float(ref["E"][-1].mean()))
    o["clamp0_maxdiff_E"] = d["clamp0_maxdiff_E"]; o["births_total"] = d["births_total"]; o["E0_mean"] = float(np.mean(E0)); o["K"] = d["K"]
    o["ref_rep_rate"] = d["ref_rep_rate"]; o["cl_rep_rate"] = d["cl_rep_rate"]; o["pop0"] = d["pop0"]
    o["valid"] = bool(d["clamp0_maxdiff_E"] == 0 and sum(d["births_total"].values()) == 0)
    return o

R = {}
for f in sorted(glob.glob(os.path.join(HERE, "runs22", "run22_*_s*.pkl"))):
    d = pickle.load(open(f, "rb")); R.setdefault(d["pop"], {})[d["seed"]] = per_seed(d)
rep = dict(per_seed=R); L = ["# Exp 22 tables (births off for the 40 targets in every arm; own-E recovery r_E; sim13 rep override)\n",
     f"50% E drop, T = {X.T}, window (T/2, T]; r_E = 1 - mean (E_ref - E_arm) / (0.5 E0), E = 0 if dead\n"]
def row(name, f, ss):
    v = [f(s) for s in ss]; m, lo, hi, n = ci(v); rep[name] = dict(seeds=ss, per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v if not np.isnan(x))))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[name]
H = "| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"
res = {}
for pop, ttl in (("true", "PRIMARY population: Exp 20 TRUE (implant) t = 1500 forks"), ("ctrl", "SECONDARY: no-implant control, Exp 17 sw_pays t = 3000 forks")) + ((("sham", "CHECK (age-matched, not a primary): Exp 19 SHAM rerun at t = 1500, implant extinct, any-agent targets"),) if "sham" in R else ()):
    P = R.get(pop, {}); ss = sorted(P); valid = bool(ss) and all(P[s]["valid"] for s in ss); rep[f"{pop}_valid_all"] = valid
    L.append(f"\n## {ttl}\n\nn = {len(ss)} seeds {ss}; clamp0 == ref exactly and 0 target births in every arm, all seeds: **{valid}** "
             f"(clamp0 max |dE| per seed {[P[s]['clamp0_maxdiff_E'] for s in ss]}; total target births per seed {[sum(P[s]['births_total'].values()) for s in ss]})\n\n" + H)
    a = row(f"{pop}: A = rE_CL - rE_OPEN", lambda s: P[s]["cl"]["rE"] - P[s]["open"]["rE"], ss); b = row(f"{pop}: B = rE_CL - rE_FROZEN", lambda s: P[s]["cl"]["rE"] - P[s]["frozen"]["rE"], ss)
    for arm in ("cl", "open", "frozen", "twin"): row(f"{pop}: rE_{arm}", lambda s, a_=arm: P[s][a_]["rE"], ss)
    row(f"{pop}: twin - ref noise |1 - rE_twin|", lambda s: abs(1 - P[s]["twin"]["rE"]), ss)
    for arm in ("ref", "cl", "open", "frozen"): row(f"{pop}: target survival at T, {arm}", lambda s, a_=arm: P[s][a_]["surv"], ss)
    for arm in ("ref", "cl", "open", "frozen"): row(f"{pop}: mean target E at T, {arm}", lambda s, a_=arm: P[s][a_]["Eend"], ss)
    row(f"{pop}: ref targets' brain rep-action > 0 fraction (suppressed)", lambda s: P[s]["ref_rep_rate"], ss); row(f"{pop}: CL targets' brain rep-action > 0 fraction", lambda s: P[s]["cl_rep_rate"], ss)
    row(f"{pop}: mean target E0", lambda s: P[s]["E0_mean"], ss)
    ps = bool(valid and a["lo"] > 0 and b["lo"] > 0); rep[f"{pop}_pass"] = ps; res[pop] = (a, b, ps)
    L.append(f"\n**{pop}: A and B CIs both > 0, validity exact: {'PASS' if ps else 'FAIL'}** (A lo {a['lo']:+.3f}, B lo {b['lo']:+.3f})\n")
PT, PC = R.get("true", {}), R.get("ctrl", {}); pair = [s for s in sorted(PT) if s in PC and PT[s]["valid"] and PC[s]["valid"]]; rep["paired_seeds"] = pair
L.append(f"\n## SECONDARY: TRUE minus control, paired by seed {pair} (n = {len(pair)})\n\n" + H)
dA = row("dA = A_true - A_ctrl", lambda s: (PT[s]["cl"]["rE"] - PT[s]["open"]["rE"]) - (PC[s]["cl"]["rE"] - PC[s]["open"]["rE"]), pair)
dB = row("dB = B_true - B_ctrl", lambda s: (PT[s]["cl"]["rE"] - PT[s]["frozen"]["rE"]) - (PC[s]["cl"]["rE"] - PC[s]["frozen"]["rE"]), pair)
for arm in ("cl", "open", "frozen"): row(f"context: rE_{arm}, true - ctrl", lambda s, a_=arm: PT[s][a_]["rE"] - PC[s][a_]["rE"], pair)
row("context: mean target E0, true - ctrl", lambda s: PT[s]["E0_mean"] - PC[s]["E0_mean"], pair)
PS = R.get("sham", {}); pair_s = [s for s in sorted(PT) if s in PS and PT[s]["valid"] and PS[s]["valid"]]; rep["paired_seeds_sham"] = pair_s
if pair_s:
    L.append(f"\n## CHECK: TRUE minus age-matched SHAM (t = 1500), paired by seed {pair_s} (n = {len(pair_s)})\n\n" + H)
    row("check: dA = A_true - A_sham", lambda s: (PT[s]["cl"]["rE"] - PT[s]["open"]["rE"]) - (PS[s]["cl"]["rE"] - PS[s]["open"]["rE"]), pair_s)
    row("check: dB = B_true - B_sham", lambda s: (PT[s]["cl"]["rE"] - PT[s]["frozen"]["rE"]) - (PS[s]["cl"]["rE"] - PS[s]["frozen"]["rE"]), pair_s)
    row("check: mean target E0, true - sham", lambda s: PT[s]["E0_mean"] - PS[s]["E0_mean"], pair_s)
p1 = res.get("true", (None, None, False))[2]; rep["primary_pass"] = p1
v = "ACTIVE FORAGING RECOVERY (primary PASS)" if p1 else "NO ACTIVE FORAGING RECOVERY (primary FAIL)"
rep["verdict"] = v; L.append(f"\n**Pre-registered primary (Exp 20 TRUE population): {v}**\n")
open(os.path.join(O, "tables22.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report22.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for j, (pop, ttl) in enumerate((("true", "Exp 22 TRUE (implant) t=1500, births off: r_E"), ("ctrl", "Exp 22 no-implant control, births off: r_E"))):
    P = R.get(pop, {}); ss = sorted(P)
    for i, arm in enumerate(("twin", "cl", "open", "frozen")):
        vv = [P[s][arm]["rE"] for s in ss]; ax[j].scatter(np.full(len(vv), i) + np.linspace(-.1, .1, len(vv)), vv, s=14); ax[j].plot([i - .25, i + .25], [np.mean(vv)] * 2, "k")
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(["twin (noise)", "CL", "OPEN", "FROZEN"]); ax[j].set_title(ttl, fontsize=9); ax[j].axhline(0, c="gray", lw=.5); ax[j].axhline(1, c="gray", lw=.5, ls=":")
for i, (nm, a) in enumerate((("A", "open"), ("B", "frozen"))):
    for k, (P, lab, col) in enumerate(((PT, "TRUE implant", "C3"), (PC, "no implant", "C0"))):
        vv = [P[s]["cl"]["rE"] - P[s][a]["rE"] for s in pair]; x = 2 * i + k
        ax[2].scatter(np.full(len(vv), x) + np.linspace(-.1, .1, len(vv)), vv, s=14, c=col, label=lab if i == 0 else None); m, lo, hi, _ = ci(vv); ax[2].errorbar(x + .25, m, [[m - lo], [hi - m]], fmt="ks", ms=4)
    for s in pair: ax[2].plot([2 * i, 2 * i + 1], [PT[s]["cl"]["rE"] - PT[s][a]["rE"], PC[s]["cl"]["rE"] - PC[s][a]["rE"]], c="gray", lw=.4)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["A true", "A ctrl", "B true", "B ctrl"]); ax[2].axhline(0, c="gray", lw=.5); ax[2].legend(fontsize=8); ax[2].set_title("A, B (own E, births off) by seed, 95% t CI", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(O, "exp22_recovery.png"), dpi=110); print("\n".join(L))
