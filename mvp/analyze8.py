"""Analyze Exp 8 (threshold sweep + dynamic CF_own). Writes out8/report8.json, tables8.md, figures.
95% t-CIs over n=3 seeds. Verdict must NOT claim agency."""
import os, pickle, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs8"); O = os.path.join(HERE, "out8"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
COL = dict(sw_full="k", sw_pe="tab:blue", sw_pecost="tab:red", sw_pecost_pe="tab:purple",
           pecost_pe_static="tab:orange", sw_pecost_pe_shuffled="tab:gray", sw_no_cost="tab:olive")
MAIN = "sw_pecost_pe"
PECOST_DEP = ["sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled"]
RES_GRID = [25, 50, 100, 200]
SPD_GRID = [0.1, 0.2, 0.4, 0.5]
DEFAULT_KEY = "res100_spd0.2"
EXTREME_LOOSE = "res25_spd0.5"
EXTREME_TIGHT = "res200_spd0.1"

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(m), float(m - h), float(m + h), n)

f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]" if np.isfinite(c[1]) else f"{c[0]:+.4f} [n={c[3]}]"

runs = {}
for v in V:
    for s in (0, 1, 2):
        p = os.path.join(R, f"cf_{v}_s{s}.pkl")
        if not os.path.exists(p): continue
        runs.setdefault(v, {})[s] = pickle.load(open(p, "rb"))

rep = dict(runs={}, sweep={}, cf={}, verdict={}, meta=dict(
    n_variants=len(V), n_seeds=3, fork_t=2500, T=100,
    own_proxy=("Forks store scalar `own` (decayed local deposit EMA), not a lifetime deposit map. "
               "cf_own subtracts own from Wl at the agent's cell. Dynamic CF with own proxy, not a recovered lifetime field."),
    thresholds=dict(default=DEFAULT_KEY, extreme_loose=EXTREME_LOOSE, extreme_tight=EXTREME_TIGHT,
                    relax="res50_spd0.4", res_grid=RES_GRID, spd_grid=SPD_GRID),
))

L = ["# Experiment 8: threshold sweep + dynamic CF_own — tables\n",
     "Dark-rooming RQ only; **not** an agency claim. "
     "Forks lack lifetime deposit maps; `own` is a decayed local deposit EMA. "
     "95% t-CI over seeds (n=3 → df=2).\n"]

# ---- sweep aggregation
for v in V:
    if v not in runs: continue
    seed_rows = []
    for s in sorted(runs[v]):
        r = runs[v][s]
        seed_rows.append(dict(seed=s, n_alive=r["n_alive"], n_targets=r["n_targets"],
                              sweep={k: dict(self_built=sv["self_built"], other_built=sv["other_built"],
                                             natural=sv["natural"], dark_any=sv["dark_any"],
                                             regime_frac=sv["regime_frac"]) for k, sv in r["sweep"].items()}))
    rep["runs"][v] = seed_rows
    # per-threshold CIs
    thr_block = {}
    keys = sorted(runs[v][sorted(runs[v])[0]]["sweep"].keys())
    for key in keys:
        acc = {m: [] for m in ("self_built", "other_built", "natural", "dark_any", "explore", "settled_noisy")}
        for s in sorted(runs[v]):
            fr = runs[v][s]["sweep"][key]["regime_frac"]
            acc["self_built"].append(fr["dark: self-built"])
            acc["other_built"].append(fr["dark: other-built"])
            acc["natural"].append(fr["dark: natural"])
            acc["dark_any"].append(runs[v][s]["sweep"][key]["dark_any"])
            acc["explore"].append(fr["explore"])
            acc["settled_noisy"].append(fr["settled-noisy"])
        thr_block[key] = {m: ci(acc[m]) for m in acc}
    rep["sweep"][v] = thr_block

# ---- CF aggregation (default + relax thresholds)
for thr in ("default", "relax"):
    rep["cf"][thr] = {}
    for v in V:
        if v not in runs: continue
        acc = {k: [] for k in (
            "ref_leave", "own_leave", "d_leave", "ref_darkish", "own_darkish", "d_darkish",
            "d_res", "d_spd", "d_surp", "n_was_dark", "n_was_other",
            "n_reclass_self_beh", "n_reclass_other", "frac_reclass_dark", "frac_reclass_other",
            "cell_leave", "d_leave_cell")}
        for s in sorted(runs[v]):
            sc = runs[v][s]["scores"].get(thr, {})
            po = sc.get("paired_own", {}) or {}
            pc = sc.get("paired_cell", {}) or {}
            acc["ref_leave"].append(po.get("ref_leave_rate", np.nan))
            acc["own_leave"].append(po.get("own_leave_rate", np.nan))
            acc["d_leave"].append(po.get("d_own_leave_mean", np.nan))
            acc["ref_darkish"].append(po.get("ref_darkish_rate", np.nan))
            acc["own_darkish"].append(po.get("own_darkish_rate", np.nan))
            acc["d_darkish"].append(po.get("d_own_darkish_mean", np.nan))
            acc["d_res"].append(po.get("d_own_res_mean", np.nan))
            acc["d_spd"].append(po.get("d_own_spd_mean", np.nan))
            acc["d_surp"].append(po.get("d_own_surp_mean", np.nan))
            acc["n_was_dark"].append(sc.get("n_was_dark", 0))
            acc["n_was_other"].append(sc.get("n_was_other", 0))
            acc["n_reclass_self_beh"].append(sc.get("n_reclass_self_beh", 0))
            acc["n_reclass_other"].append(sc.get("n_reclass_other_to_self_beh", 0))
            acc["frac_reclass_dark"].append(sc.get("frac_reclass_among_dark", np.nan))
            acc["frac_reclass_other"].append(sc.get("frac_reclass_among_other", np.nan))
            acc["cell_leave"].append(pc.get("cell_leave_rate", np.nan))
            acc["d_leave_cell"].append(pc.get("d_cell_leave_mean", np.nan))
        rep["cf"][thr][v] = {k: ci(acc[k]) for k in acc}

# ---- verdict
# (i) is Exp7 self-built null threshold-fragile?
fragile = {}
for v in PECOST_DEP + ["sw_full", "sw_no_cost", "sw_pe"]:
    if v not in rep["sweep"]: continue
    d0 = rep["sweep"][v].get(DEFAULT_KEY, {}).get("self_built", (0,)*4)
    dL = rep["sweep"][v].get(EXTREME_LOOSE, {}).get("self_built", (0,)*4)
    fragile[v] = dict(default=d0, extreme_loose=dL,
                      invents_self_built=bool(np.isfinite(dL[0]) and dL[0] > 0 and (not np.isfinite(d0[0]) or d0[0] <= 0 or dL[0] > d0[0] + 1e-12)))
pecost_loose_pos = []
for v in PECOST_DEP:
    if v not in runs: continue
    vals = [runs[v][s]["sweep"][EXTREME_LOOSE]["self_built"] for s in sorted(runs[v])]
    pecost_loose_pos.append(dict(variant=v, mean=float(np.mean(vals)), seeds_pos=int(sum(x > 0 for x in vals)), vals=[float(x) for x in vals]))
any_invent = any(p["mean"] > 0 and p["seeds_pos"] > 0 for p in pecost_loose_pos)
max_loose = max((p["mean"] for p in pecost_loose_pos), default=0.0)
# qualitative: fragile only if loose mean is meaningfully above default (e.g. >0.01), else "tiny/near-null"
qual_fragile = any_invent and max_loose >= 0.01
if not any_invent:
    thr_statement = ("Under pe_cost+depletion, relaxing thresholds to res≥25 & spd<0.5 v_term does **not** invent "
                     "self-built dark rooms (mean remains 0 / no seed shows any). Exp 7's self-built null is not threshold-fragile on this grid.")
elif qual_fragile:
    thr_statement = (f"Under pe_cost+depletion, the loosest thresholds (res≥25, spd<0.5 v_term) invent non-trivial self-built "
                     f"fractions (max mean across pe_cost arms ≈ {max_loose:.4f}). Exp 7's self-built null is threshold-fragile at the extreme.")
else:
    thr_statement = (f"Under pe_cost+depletion, the loosest thresholds invent only tiny self-built fractions "
                     f"(max mean across pe_cost arms ≈ {max_loose:.4f}, a handful of agents). Default (100/0.2) stays at 0. "
                     "Exp 7's self-built null is not qualitatively threshold-fragile — relaxing invents noise-scale counts, not a regime flip.")

# (ii) does dynamic CF_own reclassify other-built as self-built (esp sw_no_cost)?
cf_def = rep["cf"].get("default", {})
reclass_stmt = []
for v in V:
    if v not in cf_def: continue
    fr = cf_def[v]["frac_reclass_other"]
    n_other = cf_def[v]["n_was_other"]
    reclass_stmt.append(dict(variant=v, frac_reclass_other=fr, n_was_other=n_other,
                             d_leave=cf_def[v]["d_leave"]))
# blunt CF summary
noc = cf_def.get("sw_no_cost", {})
main = cf_def.get(MAIN, {})
cf_statement = (
    f"Dynamic CF_own (subtract scalar `own` from Wl at cell; not a lifetime deposit field) "
    f"Δleave (own−ref) main arm {MAIN}: {f(main['d_leave']) if main else 'n/a'}; "
    f"sw_no_cost: {f(noc['d_leave']) if noc else 'n/a'}. "
    f"Frac of fork-other-built reclassified behaviourally (CF_own leaves, ref stays): "
    f"{MAIN} {f(main['frac_reclass_other']) if main else 'n/a'}, "
    f"sw_no_cost {f(noc['frac_reclass_other']) if noc else 'n/a'}."
)

rep["verdict"] = dict(
    threshold_fragile_pecost=dict(
        any_invents_under_extreme_loose=any_invent,
        qualitatively_fragile=qual_fragile,
        max_loose_mean_pecost=max_loose,
        per_variant=fragile,
        pecost_loose=pecost_loose_pos,
        statement=thr_statement,
    ),
    cf_own_reclass=dict(
        per_variant=reclass_stmt,
        statement=cf_statement,
    ),
    honesty="Forks lack lifetime deposit maps; results use the scalar `own` proxy. Dark-rooming ≠ agency; no agency claim.",
)

# ---- tables
L += ["## A) Threshold sweep — self-built / dark_any at default vs extremes\n",
      "| variant | self-built @100/0.2 | self-built @25/0.5 | self-built @200/0.1 | dark_any @100/0.2 | dark_any @25/0.5 |",
      "|---|---|---|---|---|---|"]
for v in V:
    if v not in rep["sweep"]: continue
    s = rep["sweep"][v]
    L.append(f"| {v} | {f(s[DEFAULT_KEY]['self_built'])} | {f(s[EXTREME_LOOSE]['self_built'])} | {f(s[EXTREME_TIGHT]['self_built'])} | {f(s[DEFAULT_KEY]['dark_any'])} | {f(s[EXTREME_LOOSE]['dark_any'])} |")

L += ["\n## A) Full grid — self-built mean over seeds (rows=res_min, cols=spd_frac) per variant\n"]
for v in V:
    if v not in rep["sweep"]: continue
    L.append(f"\n### {v}\n")
    L.append("| res\\\\spd | " + " | ".join(str(sf) for sf in SPD_GRID) + " |")
    L.append("|---|" + "|".join(["---"] * len(SPD_GRID)) + "|")
    for rm in RES_GRID:
        cells = []
        for sf in SPD_GRID:
            key = f"res{rm}_spd{sf}"
            cells.append(f"{rep['sweep'][v][key]['self_built'][0]:.4f}")
        L.append(f"| {rm} | " + " | ".join(cells) + " |")

L += ["\n## B) Dynamic CF_own (default thresholds; paired CF − ref)\n",
      "| variant | n dark@fork | n other@fork | ref leave | CF_own leave | Δ leave | frac reclass among other | Δ end surp |",
      "|---|---|---|---|---|---|---|---|"]
for v in V:
    if v not in cf_def: continue
    c = cf_def[v]
    L.append(f"| {v} | {c['n_was_dark'][0]:.1f} | {c['n_was_other'][0]:.1f} | {f(c['ref_leave'])} | {f(c['own_leave'])} | {f(c['d_leave'])} | {f(c['frac_reclass_other'])} | {f(c['d_surp'])} |")

L += ["\n## B) CF_own under relaxed thresholds (res≥50, spd<0.4) — for pe_cost arms with ~0 default dark\n",
      "| variant | ref leave | CF_own leave | Δ leave | frac reclass among other |",
      "|---|---|---|---|---|"]
cf_rel = rep["cf"].get("relax", {})
for v in V:
    if v not in cf_rel: continue
    c = cf_rel[v]
    L.append(f"| {v} | {f(c['ref_leave'])} | {f(c['own_leave'])} | {f(c['d_leave'])} | {f(c['frac_reclass_other'])} |")

L += ["\n## Verdict (dark-rooming only; not an agency claim)\n",
      "### (i) Is Exp 7 self-built null threshold-fragile?",
      rep["verdict"]["threshold_fragile_pecost"]["statement"],
      "",
      "### (ii) Does dynamic CF_own reclassify other-built as self-built?",
      rep["verdict"]["cf_own_reclass"]["statement"],
      "",
      rep["verdict"]["honesty"], ""]

open(os.path.join(O, "tables8.md"), "w").write("\n".join(L) + "\n")
json.dump(rep, open(os.path.join(O, "report8.json"), "w"), indent=1,
          default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o))

# ---- figures
# 1) self-built heatmap-ish: lines over spd for each res, main arms
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
axes = axes.ravel()
for i, v in enumerate(V):
    ax = axes[i]
    if v not in rep["sweep"]:
        ax.set_visible(False); continue
    for rm in RES_GRID:
        ys, ylo, yhi = [], [], []
        for sf in SPD_GRID:
            m, lo, hi, _ = rep["sweep"][v][f"res{rm}_spd{sf}"]["self_built"]
            ys.append(m if np.isfinite(m) else 0)
            ylo.append((m - lo) if np.isfinite(lo) else 0)
            yhi.append((hi - m) if np.isfinite(hi) else 0)
        ax.errorbar(SPD_GRID, ys, yerr=[ylo, yhi], marker="o", label=f"res≥{rm}", capsize=2)
    ax.axvline(0.2, color="0.7", ls="--", lw=0.7)
    ax.set_title(v, fontsize=9); ax.set_xlabel("spd_frac"); ax.set_ylabel("self-built frac")
    ax.legend(fontsize=6)
axes[-1].set_visible(False)
fig.suptitle("Exp 8 threshold sweep: self-built fraction (95% t-CI over 3 seeds)")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp8_sweep_selfbuilt.png"), dpi=120); plt.close(fig)

# 2) dark_any at default vs loose
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(V)); w = 0.35
for j, (key, lab) in enumerate(((DEFAULT_KEY, "default 100/0.2"), (EXTREME_LOOSE, "loose 25/0.5"))):
    means, elo, ehi = [], [], []
    for v in V:
        if v not in rep["sweep"]:
            means.append(0); elo.append(0); ehi.append(0); continue
        m, lo, hi, _ = rep["sweep"][v][key]["dark_any"]
        means.append(m if np.isfinite(m) else 0)
        elo.append(max(0, m - lo) if np.isfinite(lo) else 0)
        ehi.append(max(0, hi - m) if np.isfinite(hi) else 0)
    ax.bar(x + (j - 0.5) * w, means, w, yerr=[elo, ehi], label=lab, capsize=3)
ax.set_xticks(x); ax.set_xticklabels(V, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("dark_any fraction"); ax.legend(fontsize=8)
ax.set_title("Exp 8 dark_any: default vs loosest thresholds")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp8_darkany_default_vs_loose.png"), dpi=120); plt.close(fig)

# 3) CF leave rates
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, v in enumerate(V):
    if v not in cf_def: continue
    c = cf_def[v]
    ax.errorbar(i - 0.1, c["ref_leave"][0],
                yerr=[[max(0, c["ref_leave"][0]-c["ref_leave"][1]) if np.isfinite(c["ref_leave"][1]) else 0],
                      [max(0, c["ref_leave"][2]-c["ref_leave"][0]) if np.isfinite(c["ref_leave"][2]) else 0]],
                fmt="o", color="0.4", capsize=3)
    ax.errorbar(i + 0.1, c["own_leave"][0],
                yerr=[[max(0, c["own_leave"][0]-c["own_leave"][1]) if np.isfinite(c["own_leave"][1]) else 0],
                      [max(0, c["own_leave"][2]-c["own_leave"][0]) if np.isfinite(c["own_leave"][2]) else 0]],
                fmt="s", color=COL[v], capsize=3)
ax.set_xticks(range(len(V))); ax.set_xticklabels(V, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("leave rate (not resident+quiet or dead)"); ax.set_title("Exp 8 CF: ref (○) vs CF_own (■) leave rates")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp8_cf_leave.png"), dpi=120); plt.close(fig)

# 4) reclass among other-built
fig, ax = plt.subplots(figsize=(9, 4))
for i, v in enumerate(V):
    if v not in cf_def: continue
    c = cf_def[v]["frac_reclass_other"]
    ax.errorbar(i, c[0], yerr=[[max(0, c[0]-c[1]) if np.isfinite(c[1]) else 0],
                               [max(0, c[2]-c[0]) if np.isfinite(c[2]) else 0]],
                fmt="o", color=COL[v], capsize=4)
ax.axhline(0, color="0.5", lw=0.7)
ax.set_xticks(range(len(V))); ax.set_xticklabels(V, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("frac other-built reclassed by CF_own"); ax.set_title("Exp 8: CF_own behavioural reclass of other-built (default thr)")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp8_cf_reclass.png"), dpi=120); plt.close(fig)

print(open(os.path.join(O, "tables8.md")).read())
print("wrote", O)
