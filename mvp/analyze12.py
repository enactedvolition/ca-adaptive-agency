"""Analyze Exp 12 (pre-registered seed extension of Exp 11): runs11/run11_{v}.pkl (seeds 0-2) + runs12/run12_{v}_s345|s678.pkl.
Primary (fixed in advance): sw_mg2 - sw_mg2_shuffled on Price sum S(build) over t>=1000 and on mean build over t>=1000,
paired by seed, n=9. Secondary: same on seeds 3-8 only (n=6 replication), and sw_mg2 - sw_full. 95% t CIs. No agency claim."""
import os, json, pickle, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "out12"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_mg2", "sw_mg2_shuffled"]; T0 = 1000; TERMS = ("D", "S", "T", "W", "I")
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), n=n, p_two_sided=float(stats.ttest_1samp(x, 0).pvalue))
f = lambda c: f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] (n={c['n']}, p={c['p_two_sided']:.3f})"
per = {v: {} for v in V}; maxres = 0.0
for v in V:
    srcs = [(pickle.load(open(os.path.join(HERE, "runs11", f"run11_{v}.pkl"), "rb")))]
    srcs += [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(HERE, "runs12", f"run12_{v}_s[0-9]*.pkl")))]
    for r in srcs:
        for s, s_ in zip(r["seeds"], r["series"]):
            n = np.asarray(s_["n"]); m = np.arange(len(n)) >= (T0 // 10 - 1); tt = np.asarray(s_["t"])
            d = {f"sum_{k}": float(np.sum(np.asarray(s_[f"a_{k}"])[m])) for k in TERMS}
            for k in ("build", "pop", "claim", "prem", "bcost", "intake", "own_share_here"):
                d[k] = float(np.mean(np.asarray(s_[k])[tt >= T0]))
            d["net"] = d["prem"] - d["bcost"]
            maxres = max(maxres, float(np.max(np.abs(np.asarray(s_["a_resid"])))))
            per[v][int(s)] = d
seeds = sorted(set.intersection(*[set(per[v]) for v in V])); new = [s for s in seeds if s >= 3]; old = [s for s in seeds if s < 3]
assert len(seeds) == 9, seeds
rep = dict(seeds=seeds, max_price_residual=maxres, per_seed=per, arm={}, contrasts={})
for v in V:
    rep["arm"][v] = {k: ci([per[v][s][k] for s in seeds]) for k in ("build", "sum_S", "sum_T", "sum_W", "sum_D", "net", "pop", "claim", "intake")}
for a, b in (("sw_mg2", "sw_mg2_shuffled"), ("sw_mg2", "sw_full"), ("sw_mg2_shuffled", "sw_full")):
    for lab, ss in (("all9", seeds), ("new6", new), ("old3", old)):
        rep["contrasts"][f"{a} - {b} | {lab}"] = {k: ci([per[a][s][k] - per[b][s][k] for s in ss]) for k in ("sum_S", "build", "sum_D", "net", "pop")}
P = rep["contrasts"]["sw_mg2 - sw_mg2_shuffled | all9"]
rep["primary"] = dict(S_build=P["sum_S"], mean_build=P["build"],
                      S_build_holds=bool(P["sum_S"]["lo"] > 0), mean_build_holds=bool(P["build"]["lo"] > 0),
                      replication_new6_S=rep["contrasts"]["sw_mg2 - sw_mg2_shuffled | new6"]["sum_S"])
json.dump(rep, open(os.path.join(O, "report12.json"), "w"), indent=1)
L = ["# Exp 12 tables (95% t CI over seeds; paired by seed)", "", f"Seeds: {seeds}. Max |Price residual| (build) = {maxres:.1e}.", "",
     "## Arms (n=9)", "", "| arm | mean build | ΣS build | ΣT build | ΣW build | ΣΔz̄ build | net private return | pop |", "|---|---|---|---|---|---|---|---|"]
for v in V:
    A = rep["arm"][v]; L.append(f"| {v} | " + " | ".join(f(A[k]) for k in ("build", "sum_S", "sum_T", "sum_W", "sum_D", "net", "pop")) + " |")
L += ["", "## Paired contrasts", "", "| contrast | seeds | ΣS build | mean build | ΣΔz̄ build | net return | pop |", "|---|---|---|---|---|---|---|"]
for k, C in rep["contrasts"].items():
    c, lab = k.split(" | "); L.append(f"| {c} | {lab} | " + " | ".join(f(C[q]) for q in ("sum_S", "build", "sum_D", "net", "pop")) + " |")
L += ["", "## Per seed (sw_mg2 − sw_mg2_shuffled)", "", "| seed | ΔΣS build | Δ mean build |", "|---|---|---|"]
for s in seeds: L.append(f"| {s} | {per['sw_mg2'][s]['sum_S'] - per['sw_mg2_shuffled'][s]['sum_S']:+.4f} | {per['sw_mg2'][s]['build'] - per['sw_mg2_shuffled'][s]['build']:+.4f} |")
open(os.path.join(O, "tables12.md"), "w").write("\n".join(L) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for i, (k, nm) in enumerate((("sum_S", "Price ΣS build (t≥1000)"), ("build", "mean build (t≥1000)"))):
    for j, (a, b) in enumerate((("sw_mg2", "sw_mg2_shuffled"), ("sw_mg2", "sw_full"), ("sw_mg2_shuffled", "sw_full"))):
        d = [per[a][s][k] - per[b][s][k] for s in seeds]; c = ci(d)
        ax[i].scatter(np.full(len(d), j) + np.linspace(-0.12, 0.12, len(d)), d, c=["C3" if s < 3 else "C0" for s in seeds], s=14)
        ax[i].errorbar(j + 0.25, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt="ks", capsize=3)
    ax[i].axhline(0, color="gray", lw=0.8); ax[i].set_xticks([0, 1, 2]); ax[i].set_xticklabels(["mg2−shuf", "mg2−full", "shuf−full"]); ax[i].set_title(nm, fontsize=9)
fig.suptitle("Exp 12: paired contrasts, n=9 (red = seeds 0–2 from Exp 11, blue = new seeds 3–8); 95% t CI", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(O, "exp12_contrasts.png"), dpi=130)
print(json.dumps(dict(primary=rep["primary"], max_res=maxres), indent=1))
