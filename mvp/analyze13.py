"""Analyze Exp 13 (10000-step horizon, seeds 0-8): runs13/run13_{v}_s012|s345|s678.pkl.
Pre-registered (2026-11-12): sw_mg2 - sw_mg2_shuffled mean build over t>=5000, paired by seed, n=9, 95% t CI > 0.
Secondary: Price sum S(build) over t>=5000 (same contrast), sw_mg2 - sw_full, and window-by-window trajectory.
Reproducibility: seeds 0-2 series up to t=3000 must equal runs11 (Exp 11) exactly. No agency claim."""
import os, json, pickle, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "out13"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_mg2", "sw_mg2_shuffled"]; T0 = 5000; TERMS = ("D", "S", "T", "W", "I"); WIN = [(0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000)]
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), n=n, p_two_sided=float(stats.ttest_1samp(x, 0).pvalue))
f = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] (p={c['p_two_sided']:.3f})"
per = {v: {} for v in V}; ts = {v: {} for v in V}; maxres = 0.0; repro = {}
for v in V:
    r11 = pickle.load(open(os.path.join(HERE, "runs11", f"run11_{v}.pkl"), "rb"))
    for p in sorted(glob.glob(os.path.join(HERE, "runs13", f"run13_{v}_s[0-9][0-9][0-9].pkl"))):
        r = pickle.load(open(p, "rb"))
        for s, s_ in zip(r["seeds"], r["series"]):
            tt = np.asarray(s_["t"]); pt = np.arange(len(s_["a_S"])) * 10 + 20  # Price step i covers (t-10, t], t = 20 + 10 i
            d = {f"sum_{k}": float(np.sum(np.asarray(s_[f"a_{k}"])[pt > T0])) for k in TERMS}
            for k in ("build", "pop", "claim", "prem", "bcost", "intake", "own_share_here", "wall"):
                d[k] = float(np.mean(np.asarray(s_[k])[tt >= T0]))
            d["net"] = d["prem"] - d["bcost"]
            d["win"] = {f"{a}-{b}": dict(build=float(np.mean(np.asarray(s_["build"])[(tt > a) & (tt <= b)])),
                                         sum_S=float(np.sum(np.asarray(s_["a_S"])[(pt > a) & (pt <= b)])),
                                         sum_W=float(np.sum(np.asarray(s_["a_W"])[(pt > a) & (pt <= b)])),
                                         sum_D=float(np.sum(np.asarray(s_["a_D"])[(pt > a) & (pt <= b)]))) for a, b in WIN}
            maxres = max(maxres, float(np.max(np.abs(np.asarray(s_["a_resid"])))))
            per[v][int(s)] = d; ts[v][int(s)] = (tt, np.asarray(s_["build"]), pt, np.asarray(s_["a_zbar"]))
            if s in r11["seeds"]:
                o = r11["series"][r11["seeds"].index(s)]; L = len(o["a_S"]); Lt = len(o["t"])
                repro[f"{v}_s{s}"] = bool(all(np.array_equal(np.asarray(o[k]), np.asarray(s_[k])[:(L if k.startswith(("a_", "b_", "c_", "n")) else Lt)]) for k in o))
seeds = sorted(set.intersection(*[set(per[v]) for v in V])); assert len(seeds) == 9, seeds
rep = dict(seeds=seeds, T0=T0, max_price_residual=maxres, repro_vs_exp11_t3000=repro, per_seed=per, arm={}, contrasts={}, windows={})
for v in V:
    rep["arm"][v] = {k: ci([per[v][s][k] for s in seeds]) for k in ("build", "sum_S", "sum_T", "sum_W", "sum_D", "net", "pop", "claim", "own_share_here", "wall")}
PAIRS = (("sw_mg2", "sw_mg2_shuffled"), ("sw_mg2", "sw_full"), ("sw_mg2_shuffled", "sw_full"))
for a, b in PAIRS:
    rep["contrasts"][f"{a} - {b}"] = {k: ci([per[a][s][k] - per[b][s][k] for s in seeds]) for k in ("build", "sum_S", "sum_W", "sum_D", "net", "pop", "own_share_here", "wall")}
    rep["windows"][f"{a} - {b}"] = {w: {k: ci([per[a][s]["win"][w][k] - per[b][s]["win"][w][k] for s in seeds]) for k in ("build", "sum_S", "sum_W", "sum_D")} for w in per[a][seeds[0]]["win"]}
P = rep["contrasts"]["sw_mg2 - sw_mg2_shuffled"]
rep["primary"] = dict(mean_build=P["build"], holds=bool(P["build"]["lo"] > 0), secondary_S_build=P["sum_S"])
json.dump(rep, open(os.path.join(O, "report13.json"), "w"), indent=1)
L = ["# Exp 13 tables (10000 steps; 95% t CI over seeds 0-8, paired by seed; window t>=5000 unless noted)", "",
     f"Max |Price residual| (build) = {maxres:.1e}. Seeds 0-2 series to t=3000 identical to Exp 11: {all(repro.values())} ({len(repro)} checked).", "",
     f"**Pre-registered:** sw_mg2 − sw_mg2_shuffled mean build (t≥5000) = {f(P['build'])} → {'HOLDS' if rep['primary']['holds'] else 'does not hold'}.", "",
     "## Arms", "", "| arm | mean build | ΣS build | ΣT | ΣW | ΣΔz̄ | net private return | pop | own share here | wall mass |", "|---|---|---|---|---|---|---|---|---|---|"]
for v in V:
    A = rep["arm"][v]; L.append(f"| {v} | " + " | ".join(f(A[k]) for k in ("build", "sum_S", "sum_T", "sum_W", "sum_D", "net", "pop", "own_share_here", "wall")) + " |")
L += ["", "## Paired contrasts (t≥5000)", "", "| contrast | mean build | ΣS build | ΣW build | ΣΔz̄ build | net return | pop | own share here | wall mass |", "|---|---|---|---|---|---|---|---|---|"]
for k, C in rep["contrasts"].items(): L.append(f"| {k} | " + " | ".join(f(C[q]) for q in ("build", "sum_S", "sum_W", "sum_D", "net", "pop", "own_share_here", "wall")) + " |")
L += ["", "## Window trajectory of paired contrasts", "", "| contrast | window | mean build | ΣS build | ΣW build | ΣΔz̄ build |", "|---|---|---|---|---|---|"]
for k, W in rep["windows"].items():
    for w, C in W.items(): L.append(f"| {k} | {w} | " + " | ".join(f(C[q]) for q in ("build", "sum_S", "sum_W", "sum_D")) + " |")
L += ["", "## Per seed (sw_mg2 − sw_mg2_shuffled, t≥5000)", "", "| seed | Δ mean build | ΔΣS build |", "|---|---|---|"]
for s in seeds: L.append(f"| {s} | {per['sw_mg2'][s]['build'] - per['sw_mg2_shuffled'][s]['build']:+.4f} | {per['sw_mg2'][s]['sum_S'] - per['sw_mg2_shuffled'][s]['sum_S']:+.4f} |")
open(os.path.join(O, "tables13.md"), "w").write("\n".join(L) + "\n")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for v, c in zip(V, ("gray", "C3", "C0")):
    tt = ts[v][seeds[0]][0]; B = np.stack([ts[v][s][1] for s in seeds]); k = 10
    sm = np.stack([np.convolve(b, np.ones(k) / k, mode="same") for b in B]); m = sm.mean(0); se = sm.std(0, ddof=1) / np.sqrt(len(seeds))
    ax[0].plot(tt, m, color=c, label=v, lw=1); ax[0].fill_between(tt, m - 2.31 * se, m + 2.31 * se, color=c, alpha=0.2)
    pt = ts[v][seeds[0]][2]; Z = np.stack([ts[v][s][3] for s in seeds]).mean(0); ax[1].plot(pt, Z, color=c, lw=1, label=v)
ax[0].set_title("mean build action (living agents), 95% CI band", fontsize=9); ax[0].legend(fontsize=7); ax[0].axvline(T0, color="k", lw=0.6, ls=":")
ax[1].set_title("z̄ = lifetime-mean build (Price trait), seed mean", fontsize=9); ax[1].axvline(T0, color="k", lw=0.6, ls=":")
for j, (a, b) in enumerate(PAIRS):
    W = rep["windows"][f"{a} - {b}"]; xs = np.arange(len(W)) + (j - 1) * 0.2
    m = [W[w]["build"]["mean"] for w in W]; lo = [W[w]["build"]["mean"] - W[w]["build"]["lo"] for w in W]; hi = [W[w]["build"]["hi"] - W[w]["build"]["mean"] for w in W]
    ax[2].errorbar(xs, m, yerr=[lo, hi], fmt="o", ms=3, capsize=2, label=f"{a}−{b}")
ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_xticks(range(len(WIN))); ax[2].set_xticklabels([f"{a//1000}-{b//1000}k" for a, b in WIN], fontsize=7)
ax[2].set_title("paired Δ mean build by window, 95% t CI (n=9)", fontsize=9); ax[2].legend(fontsize=6)
fig.suptitle("Exp 13: 10000-step horizon, mass-ownership guard 2 vs heredity-broken vs no guard", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(O, "exp13_trajectory.png"), dpi=130)
print(json.dumps(dict(primary=rep["primary"], max_res=maxres, repro=repro), indent=1))
