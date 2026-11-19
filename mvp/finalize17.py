"""Exp 17 tables/figures -> figures/exp17/. Compete gate (runs17/compete_thr*.json) and, if present, evolution summaries
(runs17/run17_{v}_s*.pkl: exact Price decomposition per 10-step interval, residual reported)."""
import os, json, glob, pickle, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs17"); F = os.path.join(HERE, "..", "figures", "exp17")
os.makedirs(F, exist_ok=True)
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return float(x.mean()), float(x.mean() - h), float(x.mean() + h), n
rep = dict(compete={}); md = ["# Exp 17 tables", "", "## Mixed-competition gate (aware = rep iff E > THR; C = E_endow + E_over = 0.792)", "",
      "| THR | parity | vs | dshare s0 | s1 | s2 | mean [95% CI] | all > 0 |", "|---|---|---|---|---|---|---|---|"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for f in sorted(glob.glob(os.path.join(R, "compete_thr*.json"))):
    d = json.load(open(f)); key = os.path.basename(f)[8:-5]; rep["compete"][key] = {}
    for vs in ("vs_always", "vs_blind"):
        x = [r["dshare"] for r in d[vs]]; m, lo, hi, n = ci(x)
        rep["compete"][key][vs] = dict(dshare=x, mean=m, ci=[lo, hi], all_pos=all(v > 0 for v in x),
                                       p_matched=[r.get("p_matched") for r in d[vs]], births_aware=[r["births_aware"] for r in d[vs]], births_other=[r["births_other"] for r in d[vs]])
        md.append(f"| {d['thr']} | {'odd=aware (post hoc)' if d.get('swap') else 'even=aware'} | {vs[3:]} | " + " | ".join(f"{v:+.3f}" for v in x) + f" | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {all(v > 0 for v in x)} |")
        if not d.get("swap") and d["thr"] in (0.8, 1.6):
            for r in d[vs]:
                ax[0 if vs == "vs_always" else 1].plot(np.arange(len(r["share_curve"])) * 10, r["share_curve"], ls="-" if d["thr"] == 0.8 else ":",
                                                       label=f"THR {d['thr']} s{r['seed']}")
g = rep["compete"]["thr0.8"]; rep["GATE_PASS"] = bool(g["vs_always"]["all_pos"] and g["vs_blind"]["all_pos"])
for a, t in zip(ax, ("aware vs always-reproduce", "aware vs matched-rate blind")):
    a.set_title(t); a.set_xlabel("step"); a.set_ylim(0, 1.02); a.grid(alpha=.3)
ax[0].set_ylabel("aware lineages' share"); ax[1].legend(fontsize=7, ncol=2)
fig.suptitle("Exp 17 gate: ungated fixed-cost reproduction (sim13), Exp 15b forks, 400 steps"); fig.tight_layout(); fig.savefig(os.path.join(F, "exp17_compete.png"), dpi=110)
md += ["", f"Pre-registered gate (THR 0.8, both contrasts, 3/3 seeds > 0): **{'PASS' if rep['GATE_PASS'] else 'FAIL'}**", ""]
# evolution
evo = {}
for v in ("sw_full", "sw_pays", "sw_pays_blind"):
    ser = []
    for p in sorted(glob.glob(os.path.join(R, f"run17_{v}_s*.pkl"))):
        d = pickle.load(open(p, "rb")); ser += list(zip(d["seeds"], d["series"]))
    if ser: evo[v] = sorted(ser, key=lambda x: x[0])
if evo:
    md += ["## Evolution (3000 steps; exact Price decomposition, D = S + T + W + I + resid)", "",
           "| arm | n seeds | pop (last 500) | rep–E corr (last 500) | z̄ rep-sens c_esr/l0n at 3000 | ΣS rep-sens | ΣS net intake | max abs resid |", "|---|---|---|---|---|---|---|---|"]
    rep["evo"] = {}
    for v, ser in evo.items():
        pop = [np.mean(s["pop"][s["t"] > 2500]) for _, s in ser]; cor = [np.nanmean(s["rep_E_corr"][s["t"] > 2500]) for _, s in ser]
        za = [s["a_zbar1"][-1] for _, s in ser]; sa = [np.sum(s["a_S"]) for _, s in ser]; sc = [np.sum(s["c_S"]) for _, s in ser]
        res = max(float(np.max(np.abs(s[f"{q}_resid"]))) for _, s in ser for q in "abc")
        rep["evo"][v] = dict(seeds=[sd for sd, _ in ser], pop=ci(pop), rep_E_corr=ci(cor), rep_E_corr_per_seed=cor, zbar_a=ci(za), sumS_a=ci(sa), sumS_c=ci(sc), max_resid=res)
        f2 = lambda c: f"{c[0]:+.4g} [{c[1]:+.4g}, {c[2]:+.4g}]"
        md.append(f"| {v} | {len(ser)} | {f2(ci(pop))} | {f2(ci(cor))} | {f2(ci(za))} | {f2(ci(sa))} | {f2(ci(sc))} | {res:.2e} |")
    md += ["", "Seed-matched contrasts (same seed keys; mean [95% t CI], n = 9):", ""]
    rep["evo_contrasts"] = {}
    per = {v: dict(pop=[np.mean(s["pop"][s["t"] > 2500]) for _, s in ser], cor=[np.nanmean(s["rep_E_corr"][s["t"] > 2500]) for _, s in ser],
                   za=[s["a_zbar1"][-1] for _, s in ser], sa=[np.sum(s["a_S"]) for _, s in ser], E=[np.mean(s["E"][s["t"] > 2500]) for _, s in ser],
                   rep=[np.mean(s["rep"][s["t"] > 2500]) for _, s in ser]) for v, ser in evo.items()}
    for a_, b_ in (("sw_pays", "sw_pays_blind"), ("sw_pays", "sw_full")):
        if a_ in per and b_ in per and len(per[a_]["pop"]) == len(per[b_]["pop"]):
            for k in ("pop", "cor", "za", "sa", "E", "rep"):
                c = ci(np.array(per[a_][k]) - np.array(per[b_][k])); rep["evo_contrasts"][f"{a_}-{b_}:{k}"] = c
                md.append(f"- {a_} − {b_}, {k}: {c[0]:+.4g} [{c[1]:+.4g}, {c[2]:+.4g}]")
    for v in per: md.append(f"- {v}: mean E (last 500) {ci(per[v]['E'])[0]:.3f}, rep-action>0 fraction {ci(per[v]['rep'])[0]:.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for v, ser in evo.items():
        for i, (sd, s) in enumerate(ser):
            kw = dict(color=dict(sw_full="C0", sw_pays="C3", sw_pays_blind="C2")[v], alpha=.6, label=v if i == 0 else None)
            ax[0].plot(s["t"], s["pop"], **kw); ax[1].plot(s["t"], s["rep_E_corr"], **kw); ax[2].plot(s["t"], s["E"], **kw)
    for a, t in zip(ax, ("population", "corr(rep action > 0, E)", "mean E")): a.set_title(t); a.grid(alpha=.3); a.set_xlabel("step")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(F, "exp17_evo.png"), dpi=110)
json.dump(rep, open(os.path.join(F, "report17.json"), "w"), indent=1)
open(os.path.join(F, "tables17.md"), "w").write("\n".join(md) + "\n")
for f in glob.glob(os.path.join(R, "*.json")): shutil.copy(f, F)
print("\n".join(md))
