"""Analyze Exp 7 (dark-room / niche-construction on Exp 5 forks). Writes out7/report7.json,
out7/tables7.md and figures. 95% t-CIs over n=3 seeds (df=2), same style as analyze5/6."""
import os, pickle, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import metrics2 as M2

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs7"); O = os.path.join(HERE, "out7"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
COL = dict(sw_full="k", sw_pe="tab:blue", sw_pecost="tab:red", sw_pecost_pe="tab:purple",
           pecost_pe_static="tab:orange", sw_pecost_pe_shuffled="tab:gray", sw_no_cost="tab:olive")
REG = list(M2.REGIMES)
REG_SHORT = {"explore": "explore", "settled-noisy": "settled-noisy",
             "dark: natural": "natural", "dark: other-built": "other-built", "dark: self-built": "self-built"}
MAIN = "sw_pecost_pe"
CONTRASTS = [
    ("sw_pecost_pe", "sw_full"),
    ("sw_pecost_pe", "sw_pecost_pe_shuffled"),
    ("sw_pecost_pe", "pecost_pe_static"),
    ("sw_pecost_pe", "sw_no_cost"),
]
CMETRICS = ["dark_any", "self_built", "other_built", "natural", "mean_surp", "wall_mass"]

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(m), float(m - h), float(m + h), n)

f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]" if np.isfinite(c[1]) else f"{c[0]:+.4f} [n={c[3]}]"

runs = {}
for v in V:
    for s in (0, 1, 2):
        p = os.path.join(R, f"dark_{v}_s{s}.pkl")
        if not os.path.exists(p): continue
        r = pickle.load(open(p, "rb"))
        runs.setdefault(v, {})[s] = r

rep = dict(runs={}, regimes={}, summaries={}, contrasts={}, verdict={}, meta=dict(n_variants=len(V), n_seeds=3, env_var=0.35, fork_t=2500))
L = ["# Experiment 7: dark-room / niche-construction on Exp 5 forks (step 2500) — tables\n",
     "Classification uses `metrics2.regimes` with v2 thresholds (resident = res≥100 & spd < 0.2 v_term; quiet via analytic OU variance). "
     "Fractions are over alive agents. 95% t-CI over seeds (n=3 → df=2). Dark-rooming is a separate RQ; this does **not** claim agency.\n"]

# per-variant seed arrays
for v in V:
    if v not in runs: continue
    acc = {k: [] for k in ["dark_any", "self_built", "other_built", "natural", "explore", "settled_noisy",
                            "mean_surp", "median_surp", "wall_mass", "frac_resident", "mean_W_here",
                            "mean_res", "mean_spd", "mean_own", "n_alive"]}
    for name in REG:
        acc["frac_" + REG_SHORT[name]] = []
    seed_rows = []
    for s in sorted(runs[v]):
        r = runs[v][s]; fr = r["regime_frac"]; sm = r["summaries"]
        row = dict(
            seed=s, n_alive=r["n_alive"], dark_any=r["dark_any"],
            self_built=fr["dark: self-built"], other_built=fr["dark: other-built"], natural=fr["dark: natural"],
            explore=fr["explore"], settled_noisy=fr["settled-noisy"],
            mean_surp=sm["mean_surp"], median_surp=sm["median_surp"], wall_mass=sm["wall_mass"],
            frac_resident=sm["frac_resident"], mean_W_here=sm["mean_W_here"],
            mean_res=sm["mean_res"], mean_spd=sm["mean_spd"], mean_own=sm["mean_own"],
            layout_index=r["layout_index"], regime_counts=r["regime_counts"],
        )
        seed_rows.append(row)
        for k in ("dark_any", "self_built", "other_built", "natural", "explore", "settled_noisy",
                  "mean_surp", "median_surp", "wall_mass", "frac_resident", "mean_W_here",
                  "mean_res", "mean_spd", "mean_own", "n_alive"):
            acc[k].append(row[k])
        for name in REG:
            acc["frac_" + REG_SHORT[name]].append(fr[name])
    rep["runs"][v] = seed_rows
    rep["regimes"][v] = {k: ci(acc[k]) for k in ("dark_any", "self_built", "other_built", "natural", "explore", "settled_noisy")}
    rep["regimes"][v].update({("frac_" + REG_SHORT[name]): ci(acc["frac_" + REG_SHORT[name]]) for name in REG})
    rep["summaries"][v] = {k: ci(acc[k]) for k in ("mean_surp", "median_surp", "wall_mass", "frac_resident",
                                                     "mean_W_here", "mean_res", "mean_spd", "mean_own", "n_alive")}

# paired contrasts
for a, b in CONTRASTS:
    if a not in runs or b not in runs: continue
    block = {}
    for m in CMETRICS:
        diffs = []
        for s in (0, 1, 2):
            if s not in runs[a] or s not in runs[b]: continue
            ra, rb = runs[a][s], runs[b][s]
            def get(r, key):
                if key == "dark_any": return r["dark_any"]
                if key == "self_built": return r["regime_frac"]["dark: self-built"]
                if key == "other_built": return r["regime_frac"]["dark: other-built"]
                if key == "natural": return r["regime_frac"]["dark: natural"]
                if key == "mean_surp": return r["summaries"]["mean_surp"]
                if key == "wall_mass": return r["summaries"]["wall_mass"]
                raise KeyError(key)
            diffs.append(get(ra, m) - get(rb, m))
        block[m] = dict(diff=ci(diffs), per_seed=[float(x) for x in diffs])
    rep["contrasts"][f"{a}_vs_{b}"] = block

# verdict: do self-built dark rooms appear under pe_cost with depletion on?
# pe_cost arms with depletion: sw_pecost, sw_pecost_pe, pecost_pe_static, sw_pecost_pe_shuffled
# (sw_no_cost removes metabolic cost; static still has depletion)
pecost_deplete = ["sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled"]
self_built_any = {}
for v in pecost_deplete:
    if v not in rep["regimes"]: continue
    c = rep["regimes"][v]["self_built"]
    # "appear" = mean > 0 and at least one seed > 0; report CI
    seeds_pos = sum(1 for row in rep["runs"][v] if row["self_built"] > 0)
    self_built_any[v] = dict(mean_ci=c, seeds_with_any=seeds_pos, n_seeds=c[3],
                              appears=bool(c[0] > 0 and seeds_pos > 0))
rep["verdict"] = dict(
    self_built_under_pecost_with_depletion=self_built_any,
    statement=("Self-built dark rooms appear under pe_cost with depletion still on in: "
               + (", ".join(v for v, d in self_built_any.items() if d["appears"]) or "no pe_cost+depletion arm")
               + ". This is a dark-rooming result, not an agency claim."),
)

# ---- tables
L += ["## Regime fractions (alive agents; 95% t-CI over seeds)\n",
      "| variant | seeds | n alive | explore | settled-noisy | dark: natural | dark: other-built | dark: self-built | dark_any |",
      "|---|---|---|---|---|---|---|---|---|"]
for v in V:
    if v not in rep["regimes"]: continue
    r = rep["regimes"][v]; s = rep["summaries"][v]
    L.append(f"| {v} | {r['dark_any'][3]} | {s['n_alive'][0]:.0f} | {f(r['explore'])} | {f(r['settled_noisy'])} | {f(r['natural'])} | {f(r['other_built'])} | {f(r['self_built'])} | {f(r['dark_any'])} |")

L += ["\n## Continuous summaries (alive agents)\n",
      "| variant | mean res | mean spd | mean own | mean surp | wall mass ΣWl | frac resident | mean W_here |",
      "|---|---|---|---|---|---|---|---|"]
for v in V:
    if v not in rep["summaries"]: continue
    s = rep["summaries"][v]
    L.append(f"| {v} | {f(s['mean_res'])} | {f(s['mean_spd'])} | {f(s['mean_own'])} | {f(s['mean_surp'])} | {f(s['wall_mass'])} | {f(s['frac_resident'])} | {f(s['mean_W_here'])} |")

L += ["\n## Paired contrasts by seed (main arm − control; positive = more of the metric in the main arm)\n",
      "| contrast | dark_any | self-built | other-built | natural | mean surp | wall mass |",
      "|---|---|---|---|---|---|---|"]
for a, b in CONTRASTS:
    key = f"{a}_vs_{b}"
    if key not in rep["contrasts"]: continue
    c = rep["contrasts"][key]
    L.append(f"| {a} − {b} | {f(c['dark_any']['diff'])} | {f(c['self_built']['diff'])} | {f(c['other_built']['diff'])} | {f(c['natural']['diff'])} | {f(c['mean_surp']['diff'])} | {f(c['wall_mass']['diff'])} |")

L += ["\n## Per-seed raw (main arm and key controls)\n",
      "| variant | seed | n alive | explore | settled | natural | other-built | self-built | dark_any | mean surp | wall mass |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for v in [MAIN, "sw_full", "sw_pecost", "sw_no_cost", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_pe"]:
    if v not in rep["runs"]: continue
    for row in rep["runs"][v]:
        L.append(f"| {v} | {row['seed']} | {row['n_alive']} | {row['explore']:.4f} | {row['settled_noisy']:.4f} | {row['natural']:.4f} | {row['other_built']:.4f} | {row['self_built']:.4f} | {row['dark_any']:.4f} | {row['mean_surp']:+.3f} | {row['wall_mass']:.2f} |")

L += ["\n## Verdict (dark-rooming RQ only; not an agency claim)\n", rep["verdict"]["statement"], ""]
for v, d in self_built_any.items():
    L.append(f"- {v}: self-built mean {f(d['mean_ci'])}; seeds with any self-built = {d['seeds_with_any']}/{d['n_seeds']}; appears={d['appears']}")

open(os.path.join(O, "tables7.md"), "w").write("\n".join(L) + "\n")
json.dump(rep, open(os.path.join(O, "report7.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))

# ---- figures
# 1) regime composition stacked bars
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(V)); width = 0.7
bottoms = np.zeros(len(V))
colors = {"explore": "#7df9ff", "settled-noisy": "#888888", "dark: natural": "#3d4cff",
          "dark: other-built": "#c39bff", "dark: self-built": "#ffb000"}
for name in REG:
    means = []; errs_lo = []; errs_hi = []
    for v in V:
        if v not in rep["regimes"]:
            means.append(0); errs_lo.append(0); errs_hi.append(0); continue
        key = "explore" if name == "explore" else ("settled_noisy" if name == "settled-noisy" else
              ("natural" if name == "dark: natural" else ("other_built" if name == "dark: other-built" else "self_built")))
        m, lo, hi, _ = rep["regimes"][v][key]
        means.append(m if np.isfinite(m) else 0)
        errs_lo.append(max(0, (m - lo) if np.isfinite(lo) else 0))
        errs_hi.append(max(0, (hi - m) if np.isfinite(hi) else 0))
    means = np.array(means)
    ax.bar(x, means, width, bottom=bottoms, color=colors[name], label=name, edgecolor="k", lw=0.3)
    bottoms = bottoms + means
ax.set_xticks(x); ax.set_xticklabels(V, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("fraction of alive agents"); ax.set_ylim(0, 1.05)
ax.set_title("Exp 7 regime composition on Exp 5 forks (t=2500; mean over 3 seeds)")
ax.legend(fontsize=7, loc="upper right"); fig.tight_layout()
fig.savefig(os.path.join(O, "exp7_regime_composition.png"), dpi=120); plt.close(fig)

# 2) self-built vs controls
fig, ax = plt.subplots(figsize=(8, 4))
for i, v in enumerate(V):
    if v not in rep["regimes"]: continue
    m, lo, hi, _ = rep["regimes"][v]["self_built"]
    ax.errorbar(i, m, yerr=[[max(0, m - lo) if np.isfinite(lo) else 0], [max(0, hi - m) if np.isfinite(hi) else 0]],
                fmt="o", color=COL[v], capsize=4, label=v)
    ys = [row["self_built"] for row in rep["runs"][v]]
    ax.scatter([i] * len(ys), ys, color=COL[v], s=18, alpha=0.5)
ax.axhline(0, color="0.5", lw=0.7); ax.set_xticks(range(len(V))); ax.set_xticklabels(V, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("fraction self-built dark rooms"); ax.set_title("Exp 7 self-built dark rooms (95% t-CI over 3 seeds)")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp7_selfbuilt.png"), dpi=120); plt.close(fig)

# 3) dark_any + surp/wall summaries
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j, (key, title, src) in enumerate((
    ("dark_any", "dark_any fraction", "regimes"),
    ("mean_surp", "mean surprisal", "summaries"),
    ("wall_mass", "wall mass ΣWl", "summaries"),
)):
    ax = axes[j]
    for i, v in enumerate(V):
        block = rep[src].get(v)
        if not block: continue
        m, lo, hi, _ = block[key]
        ax.errorbar(i, m, yerr=[[(m - lo) if np.isfinite(lo) else 0], [(hi - m) if np.isfinite(hi) else 0]],
                    fmt="o", color=COL[v], capsize=4)
        if src == "regimes":
            ys = [row[key] for row in rep["runs"][v]]
        else:
            ys = [row[key] for row in rep["runs"][v]]
        ax.scatter([i] * len(ys), ys, color=COL[v], s=14, alpha=0.5)
    ax.set_xticks(range(len(V))); ax.set_xticklabels(V, rotation=30, ha="right", fontsize=7)
    ax.set_title(title); ax.axhline(0, color="0.7", lw=0.5)
fig.suptitle("Exp 7 dark_any / surprisal / wall mass"); fig.tight_layout()
fig.savefig(os.path.join(O, "exp7_surp_wall.png"), dpi=120); plt.close(fig)

# 4) paired contrasts forest
fig, ax = plt.subplots(figsize=(9, 5))
ylabels = []; ypos = []
pos = 0
for a, b in CONTRASTS:
    key = f"{a}_vs_{b}"
    if key not in rep["contrasts"]: continue
    for m in CMETRICS:
        c = rep["contrasts"][key][m]["diff"]
        m_, lo, hi, _ = c
        ax.errorbar(m_, pos, xerr=[[(m_ - lo) if np.isfinite(lo) else 0], [(hi - m_) if np.isfinite(hi) else 0]],
                    fmt="o", color="tab:purple" if m == "self_built" else "0.3", capsize=3)
        ylabels.append(f"{a}−{b}: {m}"); ypos.append(pos); pos += 1
    pos += 0.5
ax.axvline(0, color="0.5", lw=0.8); ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=7)
ax.set_xlabel("paired Δ (main − control)"); ax.set_title("Exp 7 paired contrasts by seed (95% t-CI)")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp7_contrasts.png"), dpi=120); plt.close(fig)

print(open(os.path.join(O, "tables7.md")).read())
print("wrote", O)
