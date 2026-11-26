"""Exp 24 analysis: state-matched passive movement control (births off, own E). PRIMARY (Exp 20 TRUE t=1500): A_GC = r_CL - r*_GC (r*_GC: baseline = GC0, the
gradient-map policy without the drop) alongside B = r_CL - r_FROZEN. Secondary: A_GCX, r*_GC - r_MOVE, A, A_move, ctrl and sham. Validity: CL/OPEN/FROZEN == Exp 22,
MOVE == Exp 23 bit-for-bit, clamp0 == ref, 0 target births; GC0 vs ref quantified (max |dE|, r_E(GC0 vs ref), thrust-fit R^2). Writes figures/exp24/{tables24.md, report24.json, exp24_gc.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp24"); os.makedirs(O, exist_ok=True)
import exp20_run as X
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean() if n else float("nan")
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
def rE(a, b, E0): return float(X.recovery(a["E"], a["passed"], b["E"], b["passed"], E0, offspring=False).mean())   # arm a vs baseline b
def per_seed(d):
    A = d["arms"]; ref = A["ref"]; E0 = d["E0"]; o = {}
    for arm in ("cl", "open", "frozen", "move", "gc", "gcx"): o[arm] = rE(A[arm], ref, E0)
    o["twin"] = rE(d["twin"], ref, E0)
    for k in ("gc", "gcx"):
        o[k + "_star"] = rE(A[k], A[k + "0"], E0); o[k + "0_vs_ref"] = rE(A[k + "0"], ref, E0)
        o[k + "_r2"] = float(np.nanmedian(d["fit"][k]["r2"])); o[k + "_pooled"] = int(d["fit"][k]["pooled"].sum()); o[k + "_hook"] = d["fit"][k]["hook_maxdiff"]
        o[k + "0_maxdE"] = d["gc0_maxdiff_E"][k]
    o["surv"] = {a: float(A[a]["alive"][-1].mean()) for a in ("ref", "cl", "open", "frozen", "move", "gc", "gc0", "gcx", "gcx0")}
    o["valid"] = bool(d["ref_matches_exp22"] and all(d["match22"].values()) and all(d["match23"].values()) and d["clamp0_maxdiff_E"] == 0 and sum(d["births_total"].values()) == 0)
    o["detail"] = dict(ref=d["ref_matches_exp22"], m22=d["match22"], m23=d["match23"], clamp0=d["clamp0_maxdiff_E"], births=sum(d["births_total"].values())); return o
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "runs24", "run24_*_s*.pkl"))):
    d = pickle.load(open(f, "rb")); R.setdefault(d["pop"], {})[d["seed"]] = per_seed(d)
rep = dict(per_seed=R); L = ["# Exp 24 tables (births off; own-E r_E; GC = thrust from a fixed energy-blind map of the current sensed resource gradient, fitted on the target's ref run; r*_GC baseline = GC0 (same policy, no drop))\n"]
H = "| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"
def row(name, f, ss):
    v = [f(s) for s in ss]; m, lo, hi, n = ci(v); rep[name] = dict(seeds=ss, per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v)))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[name]
verd = {}
for pop, ttl in (("true", "PRIMARY: Exp 20 TRUE (implant) t = 1500 forks"), ("ctrl", "SECONDARY: no-implant control (Exp 17 sw_pays t = 3000)"), ("sham", "SECONDARY: age-matched SHAM t = 1500")):
    P = R.get(pop, {}); ss = sorted(P)
    if not ss: continue
    valid = all(P[s]["valid"] for s in ss); rep[f"{pop}_valid_all"] = valid
    L.append(f"\n## {ttl}\n\nn = {len(ss)} seeds {ss}. Validity (CL/OPEN/FROZEN == Exp 22 and MOVE == Exp 23 bit-for-bit; clamp0 == ref; 0 target births): **{valid}**"
             + ("" if valid else f"  INVALID seeds: {[(s, P[s]['detail']) for s in ss if not P[s]['valid']]}")
             + f"; max hook |thrust - tanh(x.tmap)| over seeds: gc {max(P[s]['gc_hook'] for s in ss):.2e}, gcx {max(P[s]['gcx_hook'] for s in ss):.2e}\n\n" + H)
    Ag = row(f"{pop}: A_GC = r_CL - r*_GC (PRIMARY)", lambda s: P[s]["cl"] - P[s]["gc_star"], ss)
    B = row(f"{pop}: B = r_CL - r_FROZEN", lambda s: P[s]["cl"] - P[s]["frozen"], ss)
    row(f"{pop}: A_GCX = r_CL - r*_GCX", lambda s: P[s]["cl"] - P[s]["gcx_star"], ss)
    row(f"{pop}: r*_GC - r_MOVE (A_move removed by state-matched control)", lambda s: P[s]["gc_star"] - P[s]["move"], ss)
    row(f"{pop}: r*_GCX - r_MOVE", lambda s: P[s]["gcx_star"] - P[s]["move"], ss)
    row(f"{pop}: A = r_CL - r_OPEN (Exp 22)", lambda s: P[s]["cl"] - P[s]["open"], ss)
    row(f"{pop}: A_move = r_CL - r_MOVE (Exp 23)", lambda s: P[s]["cl"] - P[s]["move"], ss)
    row(f"{pop}: A_GC raw = r_CL - r_GC (vs ref)", lambda s: P[s]["cl"] - P[s]["gc"], ss)
    for k in ("cl", "open", "frozen", "move", "gc_star", "gc", "gcx_star", "gcx", "twin"): row(f"{pop}: rE_{k}", lambda s, k=k: P[s][k], ss)
    row(f"{pop}: map cost rE(GC0 vs ref)", lambda s: P[s]["gc0_vs_ref"], ss); row(f"{pop}: map cost rE(GCX0 vs ref)", lambda s: P[s]["gcx0_vs_ref"], ss)
    row(f"{pop}: max |E_GC0 - E_ref|", lambda s: P[s]["gc0_maxdE"], ss); row(f"{pop}: max |E_GCX0 - E_ref|", lambda s: P[s]["gcx0_maxdE"], ss)
    g2 = row(f"{pop}: thrust-fit median R^2, GC (ref)", lambda s: P[s]["gc_r2"], ss); row(f"{pop}: thrust-fit median R^2, GCX (ref)", lambda s: P[s]["gcx_r2"], ss)
    row(f"{pop}: targets on pooled map, GC", lambda s: P[s]["gc_pooled"], ss)
    for a in ("ref", "cl", "open", "frozen", "move", "gc", "gc0", "gcx", "gcx0"): row(f"{pop}: target survival at T, {a}", lambda s, a=a: P[s]["surv"][a], ss)
    pos = lambda c: c["lo"] > 0
    if not valid: v = "VOID (validity failed)"
    elif pos(Ag) and pos(B): v = "ACTIVE (energy-conditioned) recovery beats passive recovery (outcome 1)"
    elif pos(Ag): v = "CL beats the state-matched passive movement control, but not via the energy sense (B CI includes 0): reactive, not agency (outcome 2)"
    else: v = "A_GC CI includes 0: no closed-loop recovery beyond a state-matched passive movement control; Exp 23 A_move consistent with stale-replay artifact (outcome 3)"
    if valid and g2["mean"] < 0.1 and pos(Ag): v += " [WEAK: GC thrust-fit median R^2 < 0.1]"
    verd[pop] = v; rep[f"{pop}_verdict"] = v; L.append(f"\n**{pop} verdict (pre-registered rule): {v}**\n")
rep["verdict"] = verd.get("true", "no true"); L.append(f"\n**Pre-registered primary verdict (TRUE): {rep['verdict']}**\n\nNo evolutionary run in Exp 24, so no Price decomposition applies.\n")
open(os.path.join(O, "tables24.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report24.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True); nm = ["A (OPEN)", "A_move", "A_GC", "A_GCX", "B (FROZEN)"]
for j, pop in enumerate(("true", "ctrl", "sham")):
    P = R.get(pop, {}); ss = sorted(P)
    if not ss: continue
    cols = [lambda s: P[s]["cl"] - P[s]["open"], lambda s: P[s]["cl"] - P[s]["move"], lambda s: P[s]["cl"] - P[s]["gc_star"], lambda s: P[s]["cl"] - P[s]["gcx_star"], lambda s: P[s]["cl"] - P[s]["frozen"]]
    for i, c in enumerate(cols):
        v = [c(s) for s in ss]; ax[j].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=14); m, lo, hi, _ = ci(v); ax[j].errorbar(i + .3, m, [[m - lo], [hi - m]], fmt="ks", ms=4)
    ax[j].axhline(0, c="gray", lw=.5); ax[j].set_xticks(range(5)); ax[j].set_xticklabels(nm, fontsize=8); ax[j].set_title(f"Exp 24 {pop}: r_CL - r_X (births off, own E)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(O, "exp24_gc.png"), dpi=110); print("\n".join(L))
