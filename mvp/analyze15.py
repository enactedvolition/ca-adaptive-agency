"""Analyze Exp 15b (NIGHTLY_LOG 2026-11-16): baseline selection on energy-sensitivity of the policy in the unchanged (sim9-equivalent) world.
runs15/run15_{sw_full,sw_full_shuffled}_s{012,345,678}.pkl. PRIMARY (fixed before any run): exact Price sum S (t>=1000) of trait a = lifetime-mean
es = d eat / d sensed energy, in sw_full, n=9 paired seeds, two-sided 95% t CI excluding 0. Secondary: sum S of |es| (b) and of net intake (c);
sw_full - sw_full_shuffled on mean es / |es| / corr(eat, E) over t=2500-3000; Price residuals. Check: sw_full pop series (every 50 steps) equals
the Exp 13 sw_full series (runs13) for seeds 0-8 bit for bit. No agency claim."""
import os, json, pickle, glob, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "out15"); os.makedirs(O, exist_ok=True)
R13 = os.environ.get("EXP13_RUNS", "/workspace/ca-agency-viz/mvp/runs13")
V = ["sw_full", "sw_full_shuffled"]; T0 = 1000; TERMS = ("D", "S", "T", "W", "I")
def ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), n=n, p_two_sided=float(stats.ttest_1samp(x, 0).pvalue))
f = lambda c: f"{c['mean']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}] (n={c['n']}, p={c['p_two_sided']:.3f})"
per = {v: {} for v in V}; maxres = 0.0; popcheck = {}
for v in V:
    for p in sorted(glob.glob(os.path.join(HERE, "runs15", f"run15_{v}_s[0-9]*.pkl"))):
        r = pickle.load(open(p, "rb"))
        for s, s_ in zip(r["seeds"], r["series"]):
            n = np.asarray(s_["n"]); m = np.arange(len(n)) >= (T0 // 10 - 1); tt = np.asarray(s_["t"])
            d = {}
            for q, nm in (("a", "es"), ("b", "abs_es"), ("c", "net")):
                for k in TERMS: d[f"{nm}_sum_{k}"] = float(np.sum(np.asarray(s_[f"{q}_{k}"])[m]))
                maxres = max(maxres, float(np.max(np.abs(np.asarray(s_[f"{q}_resid"])))))
            late = tt >= 2500
            for k in ("eat", "E", "pop", "es", "net", "eat_E_corr"): d[k + "_late"] = float(np.nanmean(np.asarray(s_[k])[late]))
            d["pop_all"] = float(np.mean(np.asarray(s_["pop"]))); d["es_zbar_end"] = float(np.asarray(s_["a_zbar1"])[-1]); d["abs_es_zbar_end"] = float(np.asarray(s_["b_zbar1"])[-1])
            per[v][int(s)] = d
            if v == "sw_full":
                tag = os.path.basename(p).split("_s")[-1][:-4]; b = [int(c) for c in tag].index(int(s))
                r13 = pickle.load(open(os.path.join(R13, f"run13_sw_full_s{tag}.pkl"), "rb"))["series"][b]
                a15 = np.asarray(s_["pop"]); a13 = np.asarray(r13["pop"])[:len(a15)]; tt13 = np.asarray(r13["t"])[:len(a15)]
                popcheck[int(s)] = bool(np.array_equal(a15, a13) and np.array_equal(tt, tt13))
seeds = sorted(set.intersection(*[set(per[v]) for v in V])); print("seeds", seeds, flush=True)
if len(seeds) < 9: print("INCOMPLETE: only", len(seeds), "paired seeds"); sys.exit(2)
rep = dict(seeds=seeds, max_price_residual=maxres, pop_series_equals_exp13_sw_full=popcheck, arm={}, contrasts={}, per_seed=per)
keys = [f"{nm}_sum_{k}" for nm in ("es", "abs_es", "net") for k in ("S", "D", "T", "W", "I")] + [k + "_late" for k in ("eat", "E", "pop", "es", "net", "eat_E_corr")] + ["es_zbar_end", "abs_es_zbar_end", "pop_all"]
for v in V: rep["arm"][v] = {k: ci([per[v][s][k] for s in seeds]) for k in keys}
rep["contrasts"]["sw_full - sw_full_shuffled"] = {k: ci([per["sw_full"][s][k] - per["sw_full_shuffled"][s][k] for s in seeds]) for k in keys}
P = rep["arm"]["sw_full"]["es_sum_S"]
rep["primary"] = dict(es_sum_S=P, excludes_zero=bool(P["lo"] > 0 or P["hi"] < 0), direction="positive" if P["mean"] > 0 else "negative")
json.dump(rep, open(os.path.join(O, "report15.json"), "w"), indent=1)
L = ["# Exp 15b tables: selection on energy-sensitivity of the policy, unchanged world (95% t CI over seeds, n = 9)", "",
     f"Max |Price residual| over all traits/seeds = {maxres:.1e}. sw_full pop series equals the Exp 13 sw_full series (t ≤ 3000) for seeds: {sorted(s for s, ok in popcheck.items() if ok)} (of {sorted(popcheck)}).", "",
     "## Pre-registered primary", "", f"- exact Price ΣS (t ≥ 1000) of lifetime-mean es = d eat / d sensed energy, sw_full: **{f(P)}** → CI {'EXCLUDES' if rep['primary']['excludes_zero'] else 'includes'} 0", "",
     "## Price terms (sum over t ≥ 1000)", "", "| arm | trait | ΣS | ΣT | ΣW | ΣI | ΣΔz̄ |", "|---|---|---|---|---|---|---|"]
for v in V:
    for nm, lab in (("es", "es"), ("abs_es", "|es|"), ("net", "net intake")):
        A = rep["arm"][v]; L.append(f"| {v} | {lab} | " + " | ".join(f(A[f"{nm}_sum_{k}"]) for k in ("S", "T", "W", "I", "D")) + " |")
L += ["", "## Late (t = 2500–3000) means", "", "| arm | pop | mean eat | mean E | mean es | mean net intake | corr(eat, E) | lifetime-mean es (pop mean, t=3000) | |es| (t=3000) |", "|---|---|---|---|---|---|---|---|---|"]
for v in V:
    A = rep["arm"][v]; L.append(f"| {v} | " + " | ".join(f(A[k]) for k in ("pop_late", "eat_late", "E_late", "es_late", "net_late", "eat_E_corr_late", "es_zbar_end", "abs_es_zbar_end")) + " |")
C = rep["contrasts"]["sw_full - sw_full_shuffled"]
L += ["", "## Paired contrast sw_full − sw_full_shuffled (heredity intact − broken)", "", "| measure | estimate |", "|---|---|"]
for k in ("es_sum_S", "abs_es_sum_S", "net_sum_S", "es_zbar_end", "abs_es_zbar_end", "es_late", "eat_E_corr_late", "pop_late"): L.append(f"| {k} | {f(C[k])} |")
open(os.path.join(O, "tables15.md"), "w").write("\n".join(L) + "\n")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for i, (k, nm) in enumerate((("es_sum_S", "ΣS of es (t≥1000)"), ("abs_es_sum_S", "ΣS of |es|"), ("net_sum_S", "ΣS of net intake"))):
    for j, v in enumerate(V):
        d = [per[v][s][k] for s in seeds]; c = ci(d)
        ax[i].scatter(np.full(len(d), j) + np.linspace(-0.12, 0.12, len(d)), d, s=14, c="C0")
        ax[i].errorbar(j + 0.25, c["mean"], yerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt="ks", capsize=3)
    ax[i].axhline(0, color="gray", lw=0.8); ax[i].set_xticks([0, 1]); ax[i].set_xticklabels(V, fontsize=7); ax[i].set_title(nm, fontsize=9)
fig.suptitle("Exp 15b: exact Price ΣS, n = 9 seeds (dots), mean ± 95% t CI", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(O, "exp15_price.png"), dpi=130)
print(json.dumps(rep["primary"], indent=1)); print("residual", maxres, "popcheck", popcheck)
