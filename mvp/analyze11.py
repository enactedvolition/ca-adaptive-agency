"""Analyze Exp 11 (mass-claim guard dose-response, sim9): runs11/{bitcheck.json, run11_*.pkl, fork11_*.npz, cf11_*.pkl}
-> out11/report11.json, tables11.md, figures. Adds the exact one-step private return (prem) vs build cost. 95% t intervals over seeds (n=3, df=2); paired-by-seed contrasts.
Dark-rooming is not agency; no agency claim."""
import os, json, pickle
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs11"); O = os.path.join(HERE, "out11"); R9 = os.path.join(HERE, "runs10")
os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_mg05", "sw_mg1", "sw_mg2", "sw_mg2_shuffled", "sw_mg2_inert"]
GUARD = V[1:]
DOSE = dict(sw_full=0.0, sw_mg05=0.5, sw_mg1=1.0, sw_mg2=2.0)
import sys; sys.path.insert(0, HERE); import sim9; C_BUILD = sim9.Cfg().c_build
SEEDS = [0, 1, 2]; T0 = 1000; D = "res100_spd0.2"
CORNERS = ["res100_spd0.2", "res25_spd0.1", "res25_spd0.5", "res200_spd0.1", "res200_spd0.5"]
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(m), float(m - h), float(m + h), n)
f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]" if np.isfinite(c[1]) else f"{c[0]:+.4f} [n={c[3]}]"
pos = lambda c: bool(np.isfinite(c[1]) and c[1] > 0); neg = lambda c: bool(np.isfinite(c[2]) and c[2] < 0)
bit = json.load(open(os.path.join(R, "bitcheck.json")))
run = {v: pickle.load(open(os.path.join(R, f"run11_{v}.pkl"), "rb")) for v in V}
cf = {v: {s: pickle.load(open(os.path.join(R, f"cf11_{v}_s{s}.pkl"), "rb")) for s in SEEDS} for v in V}
rep = dict(bitcheck=bit, fork_vs_exp10={}, price={}, ts={}, static={}, own_share={}, cf={}, contrasts={}, notes=[])
# guard=0 arms must equal the Exp 9 (sim7) forks at 2500 in every sim7 field (anc rewritten by the Price harness; feeds nothing)
for v in ("sw_full",):
    per = {}
    for s in SEEDS:
        a = np.load(os.path.join(R, f"fork11_{v}_s{s}.npz")); p9 = os.path.join(R9, f"fork10_{v}_s{s}.npz")
        if not os.path.exists(p9): per[s] = "exp10 fork missing"; continue
        b = np.load(p9)
        per[s] = [k for k in b.files if k != "anc" and not np.array_equal(a[k], b[k], equal_nan=True)]
    rep["fork_vs_exp10"][v] = dict(per_seed=per, bit_for_bit=all(isinstance(m, list) and not m for m in per.values()))
PRICE_TERMS = ("D", "S", "T", "W", "I")
for v in V:
    for s_ in run[v]["series"]:
        s_["net"] = np.asarray(s_["prem"]) - np.asarray(s_["bcost"])
# marginal private return of building at the fork: slope of lifetime mean prem on lifetime mean build (agents with >=100 steps of life)
rep["marginal"] = {}
for v in V:
    sl, rr_ = [], []
    for s in SEEDS:
        z = np.load(os.path.join(R, f"fork11_{v}_s{s}.npz")); m = z["alive"] & (z["l0n"] >= 100)
        zb = z["c_build"][m] / z["l0n"][m]; zp = z["c_prem"][m] / z["l0n"][m]
        if m.sum() > 5 and zb.std() > 0:
            sl.append(float(np.polyfit(zb, zp, 1)[0])); rr_.append(float(np.corrcoef(zb, zp)[0, 1]))
    rep["marginal"][v] = dict(slope_prem_on_build=ci(sl), corr=ci(rr_), build_cost_slope=C_BUILD)
pr_seed = {}
for v in V:
    rr = run[v]["series"]; pr = {}; pr_seed[v] = {}
    for q, name in (("a", "build"), ("b", "claim"), ("c", "prem")):
        sums = {k: [] for k in PRICE_TERMS}; res = []
        for s_ in rr:
            n = np.asarray(s_["n"]); m = np.arange(len(n)) >= (T0 // 10 - 1)
            for k in PRICE_TERMS: sums[k].append(float(np.sum(np.asarray(s_[f"{q}_{k}"])[m])))
            res.append(float(np.max(np.abs(np.asarray(s_[f"{q}_resid"])))))
        pr[name] = {f"sum_{k}": ci(sums[k]) for k in PRICE_TERMS}
        pr[name]["max_abs_residual"] = max(res); pr[name]["zbar_end"] = ci([s_[f"{q}_zbar1"][-1] for s_ in rr])
        pr_seed[v][name] = sums
    rep["price"][v] = pr
    ts = {}
    for k in ("pop", "build", "claim", "intake", "E", "wall", "live_frac", "own_share_here", "pe", "prem", "bcost", "own_here", "net"):
        ts[k] = ci([float(np.mean(np.asarray(s_[k])[np.asarray(s_["t"]) >= T0])) for s_ in rr])
    rep["ts"][v] = ts
    c = cf[v]; st = {}
    for key in CORNERS:
        st[key] = {q: ci([c[s]["static"][key]["exact"][q] for s in SEEDS]) for q in ("self_built", "other_built", "natural", "dark_any")}
        st[key]["n_self_total"] = int(sum(c[s]["static"][key]["exact"]["regime_counts"]["dark: self-built"] for s in SEEDS))
    rep["static"][v] = st
    rep["own_share"][v] = {k: ci([c[s]["own_share"][k] for s in SEEDS]) for k in ("dark_mean_exact_share", "all_mean_exact_share", "n_dark", "frac_wall_mass_by_living", "lifetime_mass_alive_mean")}
    sc = [c[s]["scores"]["default"] for s in SEEDS]
    d = {k: ci([x[k] for x in sc]) for k in ("d_true", "d_ctrl", "d_true_ctrl", "ref_leave", "true_leave", "ctrl_leave", "d_intake_true_ctrl", "d_E_true_ctrl", "d_claim_true_ctrl", "frac_beh_self_among_dark")}
    d["n_was_dark_total"] = int(sum(x["n_was_dark"] for x in sc)); d["n_beh_self_total"] = int(sum(x["n_beh_self"] for x in sc))
    mt = np.concatenate([c[s]["targets"]["mass_true"] for s in SEEDS]); mc = np.concatenate([c[s]["targets"]["mass_ctrl"] for s in SEEDS])
    d["ctrl_mass_ratio_min"] = float(np.min(mc / np.maximum(mt, 1e-12))) if len(mt) else float("nan")
    d["n_targets_total"] = int(len(mt))
    rep["cf"][v] = d
# paired contrasts
def ts_seed(v, k):
    return [float(np.mean(np.asarray(s_[k])[np.asarray(s_["t"]) >= T0])) for s_ in run[v]["series"]]
def pair(a, b):
    out = {}
    for k in ("build", "claim", "own_share_here", "live_frac", "intake", "pop", "wall", "prem", "net"):
        out[k] = ci(np.subtract(ts_seed(a, k), ts_seed(b, k)))
    out["price_S_build"] = ci(np.subtract(pr_seed[a]["build"]["S"], pr_seed[b]["build"]["S"]))
    out["self_built_exact"] = ci([cf[a][s]["static"][D]["exact"]["self_built"] - cf[b][s]["static"][D]["exact"]["self_built"] for s in SEEDS])
    out["dark_any"] = ci([cf[a][s]["static"][D]["exact"]["dark_any"] - cf[b][s]["static"][D]["exact"]["dark_any"] for s in SEEDS])
    out["cf_true_minus_ctrl"] = ci([cf[a][s]["scores"]["default"]["d_true_ctrl"] - cf[b][s]["scores"]["default"]["d_true_ctrl"] for s in SEEDS])
    return out
PAIRS = [("sw_mg05", "sw_full"), ("sw_mg1", "sw_full"), ("sw_mg2", "sw_full"), ("sw_mg2", "sw_mg2_shuffled"), ("sw_mg2", "sw_mg2_inert"), ("sw_mg2_inert", "sw_full")]
for a, b in PAIRS: rep["contrasts"][f"{a} - {b}"] = pair(a, b)
# verdict
C = rep["contrasts"]
build_up = [p for p in C if pos(C[p]["build"])]; build_dn = [p for p in C if neg(C[p]["build"])]
sel_build = [v for v in V if pos(rep["price"][v]["build"]["sum_S"])]; sel_build_neg = [v for v in V if neg(rep["price"][v]["build"]["sum_S"])]
self_pos = [v for v in V if pos(rep["static"][v][D]["self_built"])]
cf_pos = [v for v in V if pos(rep["cf"][v]["d_true_ctrl"])]
intake_cost = [v for v in GUARD if neg(rep["cf"][v]["d_intake_true_ctrl"])]
maxres = max(rep["price"][v][q]["max_abs_residual"] for v in V for q in ("build", "claim", "prem"))
pays = [v for v in GUARD if pos(rep["ts"][v]["net"])]
marg_pays = [v for v in GUARD if np.isfinite(rep["marginal"][v]["slope_prem_on_build"][1]) and rep["marginal"][v]["slope_prem_on_build"][1] > C_BUILD]
dose = {}
for s_i in range(3):
    xs = [DOSE[v] for v in DOSE]; ys = [ts_seed(v, "build")[s_i] for v in DOSE]; dose.setdefault("build_slope_per_guard", []).append(float(np.polyfit(xs, ys, 1)[0]))
    ys = [pr_seed[v]["build"]["S"][s_i] for v in DOSE]; dose.setdefault("S_build_slope_per_guard", []).append(float(np.polyfit(xs, ys, 1)[0]))
rep["dose"] = {k: ci(x) for k, x in dose.items()}
rep["verdict"] = dict(build_up_contrasts=build_up, build_down_contrasts=build_dn, selection_on_build_pos=sel_build, selection_on_build_neg=sel_build_neg,
                      self_built_pos_default=self_pos, cf_true_beats_ctrl=cf_pos, cf_intake_cost_of_losing_own_walls=intake_cost,
                      max_price_residual=maxres, avg_return_exceeds_cost=pays, marginal_return_exceeds_cost=marg_pays,
                      max_self_built_corner_guard=max(rep["static"][v][k]["self_built"][0] for v in GUARD for k in CORNERS))
vd = rep["verdict"]
vd["statement"] = (f"Mean build action higher (CI>0) in: {', '.join(build_up) or 'none'}; lower in: {', '.join(build_dn) or 'none'}. "
                   f"Price selection on build (Σ S over t≥{T0}) CI>0 in: {', '.join(sel_build) or 'none'}; CI<0 in: {', '.join(sel_build_neg) or 'none'}. "
                   f"Exact self-built @100/0.2 CI>0 in: {', '.join(self_pos) or 'none'} (max over corners in guard arms {vd['max_self_built_corner_guard']:.4f}). "
                   f"True CF beats equal-mass control on leave in: {', '.join(cf_pos) or 'none'}; losing own walls cuts intake (CI<0) in: {', '.join(intake_cost) or 'none'}. "
                   f"Mean private return − build cost CI>0 in: {', '.join(pays) or 'none'}; marginal return of building (slope of lifetime prem on lifetime build) CI-low > c_build={C_BUILD} in: {', '.join(marg_pays) or 'none'}. "
                   f"Dose slope of build per unit guard {f(rep['dose']['build_slope_per_guard'])}; of ΣS(build) {f(rep['dose']['S_build_slope_per_guard'])}. "
                   f"Price identity residual ≤ {maxres:.1e}.")
vd["honesty"] = "n = 3 seeds (df = 2), wide CIs. Dark-rooming ≠ agency; no agency claim."
json.dump(rep, open(os.path.join(O, "report11.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
# tables
L = ["# Exp 11 tables (95% t CI over 3 seeds)", "", "## Time averages, t ≥ 1000", "", "| arm | pop | build | claim | own share here | living-owned wall frac | intake | wall mass |", "|---|---|---|---|---|---|---|---|"]
for v in V:
    t = rep["ts"][v]; L.append(f"| {v} | {f(t['pop'])} | {f(t['build'])} | {f(t['claim'])} | {f(t['own_share_here'])} | {f(t['live_frac'])} | {f(t['intake'])} | {f(t['wall'])} |")
L += ["", "## Private return vs build cost, t ≥ 1000 (energy/step, population means)", "", "| arm | prem (exact 1-step) | build cost | net | own deposit here | marginal slope prem/build (vs c_build) | corr |", "|---|---|---|---|---|---|---|"]
for v in V:
    t = rep["ts"][v]; mg = rep["marginal"][v]; L.append(f"| {v} | {f(t['prem'])} | {f(t['bcost'])} | {f(t['net'])} | {f(t['own_here'])} | {f(mg['slope_prem_on_build'])} (c_build {C_BUILD}) | {f(mg['corr'])} |")
L += ["", f"Dose response over guard 0/0.5/1/2 (per-seed OLS slope): build {f(rep['dose']['build_slope_per_guard'])}, ΣS build {f(rep['dose']['S_build_slope_per_guard'])}"]
L += ["", f"## Exact Price decomposition, summed over boundaries t ≥ {T0} (D = S+T+W+I)", "", "| arm | trait | ΣD | ΣS (selection) | ΣT (transmission) | ΣW (within-life) | ΣI (immigrants) | max abs residual |", "|---|---|---|---|---|---|---|---|"]
for v in V:
    for q in ("build", "claim", "prem"):
        p = rep["price"][v][q]; L.append(f"| {v} | {q} | {f(p['sum_D'])} | {f(p['sum_S'])} | {f(p['sum_T'])} | {f(p['sum_W'])} | {f(p['sum_I'])} | {p['max_abs_residual']:.1e} |")
L += ["", "## Exact static classification at step 2500", "", "| arm | corner | self-built | other-built | natural | dark_any | n self total |", "|---|---|---|---|---|---|---|"]
for v in V:
    for k in CORNERS:
        s_ = rep["static"][v][k]; L.append(f"| {v} | {k} | {f(s_['self_built'])} | {f(s_['other_built'])} | {f(s_['natural'])} | {f(s_['dark_any'])} | {s_['n_self_total']} |")
L += ["", "## True CF (T=100) at default thresholds", "", "| arm | targets | dark targets | Δleave true−ref | Δleave ctrl−ref | Δleave true−ctrl | Δintake true−ctrl | ΔE true−ctrl | ctrl/true mass min |", "|---|---|---|---|---|---|---|---|---|"]
for v in V:
    d = rep["cf"][v]; L.append(f"| {v} | {d['n_targets_total']} | {d['n_was_dark_total']} | {f(d['d_true'])} | {f(d['d_ctrl'])} | {f(d['d_true_ctrl'])} | {f(d['d_intake_true_ctrl'])} | {f(d['d_E_true_ctrl'])} | {d['ctrl_mass_ratio_min']:.3f} |")
L += ["", "## Paired-by-seed contrasts", "", "| contrast | build | claim | prem | net | own share here | Price ΣS build | self-built exact | dark_any | CF true−ctrl |", "|---|---|---|---|---|---|---|---|---|---|"]
for p, c in C.items():
    L.append(f"| {p} | {f(c['build'])} | {f(c['claim'])} | {f(c['prem'])} | {f(c['net'])} | {f(c['own_share_here'])} | {f(c['price_S_build'])} | {f(c['self_built_exact'])} | {f(c['dark_any'])} | {f(c['cf_true_minus_ctrl'])} |")
L += ["", "**Verdict:** " + vd["statement"], "", vd["honesty"]]
open(os.path.join(O, "tables11.md"), "w").write("\n".join(L) + "\n")
# figures
cols = dict(zip(V, plt.cm.tab10(np.arange(len(V)))))
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for v in V:
    for k, a in zip(("build", "claim", "prem", "net"), ax):
        X = np.stack([np.asarray(s_[k]) for s_ in run[v]["series"]]); t = np.asarray(run[v]["series"][0]["t"])
        a.plot(t, X.mean(0), color=cols[v], label=v); a.fill_between(t, X.min(0), X.max(0), color=cols[v], alpha=0.15); a.set_title(k); a.set_xlabel("step")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(O, "exp11_timeseries.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for v in V:
    X = np.stack([np.cumsum(np.asarray(s_["a_S"])) for s_ in run[v]["series"]]); ax[0].plot(X.mean(0), color=cols[v], label=v)
    X = np.stack([np.cumsum(np.asarray(s_["a_D"])) for s_ in run[v]["series"]]); ax[1].plot(X.mean(0), color=cols[v])
ax[0].set_title("cumulative Price selection term S, trait = lifetime build"); ax[1].set_title("cumulative ΔZ̄ (build)"); ax[0].legend(fontsize=7)
for a in ax: a.set_xlabel("boundary (×10 steps)")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp11_price_build.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); x = np.arange(len(V))
for j, (k, a) in enumerate((("self_built", ax[0]), ("dark_any", ax[0]))):
    m = [rep["static"][v][D][k][0] for v in V]; lo = [rep["static"][v][D][k][0] - rep["static"][v][D][k][1] for v in V]
    a.bar(x + 0.4 * j - 0.2, m, 0.4, yerr=np.nan_to_num(lo), label=k)
ax[0].set_xticks(x); ax[0].set_xticklabels(V, rotation=30, ha="right", fontsize=7); ax[0].legend(); ax[0].set_title("exact regime fractions @100/0.2")
m = [rep["cf"][v]["d_true_ctrl"][0] for v in V]; e = [rep["cf"][v]["d_true_ctrl"][0] - rep["cf"][v]["d_true_ctrl"][1] for v in V]
ax[1].bar(x, m, yerr=np.nan_to_num(e)); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xticks(x); ax[1].set_xticklabels(V, rotation=30, ha="right", fontsize=7); ax[1].set_title("CF Δleave true − equal-mass control")
fig.tight_layout(); fig.savefig(os.path.join(O, "exp11_selfbuilt_cf.png"), dpi=110); plt.close(fig)
print(vd["statement"])
