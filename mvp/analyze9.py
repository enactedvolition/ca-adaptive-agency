"""Analyze Exp 9 (exact deposit attribution): runs9/attr_*.json, runs9/bitcheck.json, runs9/cf9_*.pkl -> out9/report9.json, tables9.md, figures.
95% t intervals over seeds (n=3, df=2); paired-by-seed contrasts. Dark-rooming is not agency."""
import os, json, pickle, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("EXP9_OUT", os.path.join(HERE, "runs9")); O = os.environ.get("EXP9_FIG", os.path.join(HERE, "out9"))
os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
PE = ["sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled"]
SEEDS = [0, 1, 2]; MAIN = "sw_pecost_pe"
CORNERS = ["res100_spd0.2", "res25_spd0.1", "res25_spd0.5", "res200_spd0.1", "res200_spd0.5"]
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(m), float(m - h), float(m + h), n)
f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]" if np.isfinite(c[1]) else f"{c[0]:+.4f} [n={c[3]}]"
attr = {v: json.load(open(os.path.join(R, f"attr_{v}.json"))) for v in V if os.path.exists(os.path.join(R, f"attr_{v}.json"))}
bit = json.load(open(os.path.join(R, "bitcheck.json")))
runs = {v: {s: pickle.load(open(p, "rb")) for s in SEEDS for p in [os.path.join(R, f"cf9_{v}_s{s}.pkl")] if os.path.exists(p)} for v in V}
rep = dict(bitcheck=bit, attr={}, static={}, own_share={}, cf={"default": {}, "relax": {}}, contrasts={}, notes=[])
for v in V:
    ps = attr.get(v, {}).get("per_seed", {})
    rep["attr"][v] = dict(bit_vs_exp5_all=bool(ps) and all(r["bit_vs_exp5_fork"] for r in ps.values()) and len(ps) == 3,
                          mismatched={s: r["mismatched"] for s, r in ps.items()}, max_attr_residual=max([r["attr_residual"] for r in ps.values()] or [float("nan")]),
                          max_dead_slot=max([r["dead_slots_nonzero"] for r in ps.values()] or [float("nan")]))
    rr = runs[v]
    st = {}
    for key in rr[SEEDS[0]]["static"] if rr else []:
        st[key] = {f"{m}_{q}": ci([rr[s]["static"][key][m][q] for s in rr]) for m in ("proxy", "exact") for q in ("self_built", "other_built", "natural", "dark_any")}
        st[key]["exact_minus_proxy_self"] = ci([rr[s]["static"][key]["exact"]["self_built"] - rr[s]["static"][key]["proxy"]["self_built"] for s in rr])
        st[key]["n_self_exact_total"] = int(sum(rr[s]["static"][key]["exact"]["regime_counts"]["dark: self-built"] for s in rr))
        st[key]["n_self_proxy_total"] = int(sum(rr[s]["static"][key]["proxy"]["regime_counts"]["dark: self-built"] for s in rr))
    rep["static"][v] = st
    rep["own_share"][v] = {k: ci([rr[s]["own_share"][k] for s in rr]) for k in (rr[SEEDS[0]]["own_share"] if rr else {})}
    for thr in ("default", "relax"):
        keys = ("d_true", "d_ctrl", "d_true_ctrl", "d_true_ctrl_dark", "ref_leave", "true_leave", "ctrl_leave", "frac_beh_self_among_dark", "frac_beh_other_to_self", "n_was_dark", "n_was_other", "n_beh_self")
        if rr and all(rr[s]["scores"].get(thr) for s in rr):
            d = {k: ci([rr[s]["scores"][thr][k] for s in rr]) for k in keys}
            d["n_beh_self_total"] = int(sum(rr[s]["scores"][thr]["n_beh_self"] for s in rr))
            d["n_was_dark_total"] = int(sum(rr[s]["scores"][thr]["n_was_dark"] for s in rr))
            mt = np.concatenate([rr[s]["targets"]["mass_true"] for s in rr]); mc = np.concatenate([rr[s]["targets"]["mass_ctrl"] for s in rr])
            d["ctrl_mass_ratio_min"] = float(np.min(mc / np.maximum(mt, 1e-12))) if len(mt) else float("nan")
            d["mass_true_median"] = float(np.median(mt)) if len(mt) else float("nan")
            rep["cf"][thr][v] = d
def pair(a, b, fn):
    return ci([fn(a, s) - fn(b, s) for s in SEEDS if s in runs[a] and s in runs[b]])
sb = lambda v, s: runs[v][s]["static"]["res100_spd0.2"]["exact"]["self_built"]
da = lambda v, s: runs[v][s]["static"]["res100_spd0.2"]["exact"]["dark_any"]
tc = lambda v, s: runs[v][s]["scores"]["default"]["d_true_ctrl"] if runs[v][s]["scores"].get("default") else float("nan")
for other in ("sw_full", "sw_pecost_pe_shuffled", "pecost_pe_static", "sw_no_cost"):
    rep["contrasts"][f"{MAIN} - {other}"] = dict(self_built_exact=pair(MAIN, other, sb), dark_any=pair(MAIN, other, da), cf_true_minus_ctrl=pair(MAIN, other, tc))
# verdicts
D = "res100_spd0.2"
pos = lambda c: np.isfinite(c[1]) and c[1] > 0
exact_pos_def = [v for v in V if pos(rep["static"][v][D]["exact_self_built"])]
exact_any_corner = {v: max(rep["static"][v][k]["exact_self_built"][0] for k in CORNERS) for v in V}
cf_pos = [v for v in V if v in rep["cf"]["default"] and pos(rep["cf"]["default"][v]["d_true_ctrl"])]
noc = rep["static"]["sw_no_cost"][D]; nocf = rep["cf"]["default"].get("sw_no_cost", {})
rep["verdict"] = dict(
    exact_self_built_positive_default=exact_pos_def, exact_self_built_max_over_corners=exact_any_corner, cf_true_beats_ctrl=cf_pos,
    statement=(f"Exact attribution, default 100/0.2: self-built CI excludes 0 in {exact_pos_def or 'no variant'}; "
               f"sw_no_cost exact self-built {f(noc['exact_self_built'])} vs proxy {f(noc['proxy_self_built'])}, dark_any {f(noc['exact_dark_any'])}; "
               f"max exact self-built over corners in pe_cost arms {max(exact_any_corner[v] for v in PE):.4f}. "
               f"True CF (delete own lifetime deposits) − equal-mass control on leave: CI > 0 in {cf_pos or 'no variant'}"
               + (f"; sw_no_cost {f(nocf['d_true_ctrl'])}" if nocf else "") + "."),
    honesty="Dark-rooming ≠ agency; no agency claim. n = 3 seeds (df = 2), wide CIs.")
json.dump(rep, open(os.path.join(O, "report9.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
# tables
L = ["# Exp 9 tables (exact deposit attribution)", "", f"Bit-for-bit (sim7 vs sim5, {bit['steps']} steps, same harness): attr_off {bit['attr_off']['bit_for_bit']}, attr_on {bit['attr_on']['bit_for_bit']}; accounting residual {bit['attr_residual']:.1e}.", "",
     "| variant | step-2500 state == Exp 5 fork (3 seeds) | max ΣAdep+Aorph−Wl (rel) |", "|---|---|---|"]
L += [f"| {v} | {rep['attr'][v]['bit_vs_exp5_all']} | {rep['attr'][v]['max_attr_residual']:.1e} |" for v in V]
for k in CORNERS:
    L += ["", f"## Static classification @ {k}", "", "| variant | self-built exact | self-built proxy | other-built exact | natural | dark_any | exact−proxy self |", "|---|---|---|---|---|---|---|"]
    L += [f"| {v} | {f(rep['static'][v][k]['exact_self_built'])} | {f(rep['static'][v][k]['proxy_self_built'])} | {f(rep['static'][v][k]['exact_other_built'])} | {f(rep['static'][v][k]['exact_natural'])} | {f(rep['static'][v][k]['exact_dark_any'])} | {f(rep['static'][v][k]['exact_minus_proxy_self'])} |" for v in V]
L += ["", "## Own-deposit share of the wall at the agent's cell (dark agents, default thresholds)", "", "| variant | exact share | proxy share | max exact share | corr(exact, proxy) all alive | n dark |", "|---|---|---|---|---|---|"]
L += [f"| {v} | {f(rep['own_share'][v]['dark_mean_exact_share'])} | {f(rep['own_share'][v]['dark_mean_proxy_share'])} | {f(rep['own_share'][v]['dark_max_exact_share'])} | {f(rep['own_share'][v]['exact_vs_proxy_corr'])} | {f(rep['own_share'][v]['n_dark'])} |" for v in V]
for thr in ("default", "relax"):
    L += ["", f"## Dynamic CF ({thr} thresholds), leave fractions, paired differences", "", "| variant | Δleave true−ref | Δleave ctrl−ref | **true−ctrl** | true−ctrl (dark targets) | beh. self-built among dark (n total) | ctrl/true mass min |", "|---|---|---|---|---|---|---|"]
    L += [f"| {v} | {f(c['d_true'])} | {f(c['d_ctrl'])} | {f(c['d_true_ctrl'])} | {f(c['d_true_ctrl_dark'])} | {f(c['frac_beh_self_among_dark'])} ({c['n_beh_self_total']}/{c['n_was_dark_total']}) | {c['ctrl_mass_ratio_min']:.3f} |" for v, c in rep["cf"][thr].items()]
L += ["", "## Paired contrasts (by seed)", "", "| contrast | self-built exact | dark_any | CF true−ctrl |", "|---|---|---|---|"]
L += [f"| {k} | {f(c['self_built_exact'])} | {f(c['dark_any'])} | {f(c['cf_true_minus_ctrl'])} |" for k, c in rep["contrasts"].items()]
L += ["", rep["verdict"]["statement"], "", rep["verdict"]["honesty"]]
open(os.path.join(O, "tables9.md"), "w").write("\n".join(L) + "\n")
# figures
x = np.arange(len(V)); fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, k in zip(axs, ("res100_spd0.2", "res25_spd0.5")):
    for j, (m, col) in enumerate((("proxy", "#888"), ("exact", "#d62"))):
        c = np.array([rep["static"][v][k][f"{m}_self_built"][:3] for v in V])
        ax.bar(x + (j - .5) * .35, c[:, 0], .35, yerr=np.abs(np.nan_to_num(c[:, 1:].T - c[:, 0])), color=col, label=m, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(V, rotation=35, ha="right", fontsize=8); ax.set_title(f"self-built dark-room fraction @ {k}"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(O, "exp9_selfbuilt_exact_vs_proxy.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4.5)); cv = [v for v in V if v in rep["cf"]["default"]]; x = np.arange(len(cv))
for j, (k, col) in enumerate((("d_true", "#d62"), ("d_ctrl", "#888"), ("d_true_ctrl", "#26a"))):
    c = np.array([rep["cf"]["default"][v][k][:3] for v in cv])
    ax.bar(x + (j - 1) * .27, c[:, 0], .27, yerr=np.abs(np.nan_to_num(c[:, 1:].T - c[:, 0])), color=col, label=k, capsize=3)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels(cv, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Δ leave fraction (T=100)"); ax.set_title("True CF: delete own lifetime deposits vs equal-mass control"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(O, "exp9_cf_true_vs_ctrl.png"), dpi=110); plt.close(fig)
print(rep["verdict"]["statement"])
