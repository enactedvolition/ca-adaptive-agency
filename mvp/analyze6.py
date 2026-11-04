"""Analyze Exp 6 (regulated-variable recovery). Writes out6/tables6.md, out6/report6.json and figures.
Per run (variant x seed) we average over targets, then give 95% t-CIs over seeds (n=3, df=2)."""
import os, pickle, json, glob
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs6"); O = os.path.join(HERE, "out6"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
COL = dict(sw_full="k", sw_pe="tab:blue", sw_pecost="tab:red", sw_pecost_pe="tab:purple", pecost_pe_static="tab:orange", sw_pecost_pe_shuffled="tab:gray", sw_no_cost="tab:olive")
LATE = slice(60, 120); EARLY = slice(0, 20)
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else np.nan, np.nan, np.nan, n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return (float(m), float(m - h), float(m + h), n)
f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
rep = dict(runs={}, ed={}, hk={}, verdict={}, validation={})
curves = {}
L = ["# Experiment 6: recovery scored on regulated variables — tables (per-run means over targets; 95% t-CI over seeds, n=3 → df=2)\n"]
for v in V:
    acc = {k: [] for k in ("r_cl", "r_clamp", "r_blind", "A", "B", "dI_cl", "dI_clamp", "dI_blind", "AI", "surv_cl", "surv_clamp", "surv_blind", "surv_ref",
                            "hk01_E", "hk03_E", "tw_E", "hk01_vs_tw", "hk03_vs_tw", "hk01_I", "tw_I", "hk01_PE", "tw_PE", "tw_E_early", "hk01_E_early", "A_med", "A_win", "n_keep")}
    cv = []
    for s in (0, 1, 2):
        p = os.path.join(R, f"reg_{v}_s{s}.pkl")
        if not os.path.exists(p): continue
        r = pickle.load(open(p, "rb")); c = r["conds"]
        ref = c["ref"]; K = len(r["targets"])
        # The passive control replays the reference agent's recorded actions, which the sim zeroes on the step the reference
        # agent dies; so the clamp is only defined while the reference agent lives. Energy-drop metrics therefore use targets
        # whose reference agent survives the whole 120 steps (a condition on the unperturbed run only, same for all arms).
        keep = ref["own"][:, -1]
        v0 = c["clamp0"]; kv = keep[v0["ti"]]
        rep["validation"][f"{v}_s{s}"] = dict(clamp0_maxdiff=float(np.abs(v0["E"][kv] - ref["E"][v0["ti"][kv]]).max()) if kv.any() else float("nan"),
                                             clamp0_n=int(kv.sum()), clamp0_maxdiff_all=r["clamp0_maxdiff"], n_keep=int(keep.sum()), K=K, wall_s=r["wall_s"])
        Em = lambda d: np.where(d["own"], d["E"], 0.0)
        Im = lambda d: np.where(d["own"], d["c_int"], np.nan)
        Eref = Em(ref); Iref = Im(ref)
        D0 = r["E0"] * r["drop"]                                           # energy removed at t=0 (pre-step E)
        okD = D0 > 1e-3
        rr, dI, sv = {}, {}, {}
        for x in ("cl", "clamp", "blind"):
            d = c[f"ed_{x}"]; ti = d["ti"]
            Dx = Eref[ti] - Em(d)                                          # deficit (K, T)
            rr[x] = np.where(keep[ti], 1 - Dx[:, LATE].mean(1) / np.where(okD[ti], D0[ti], np.nan), np.nan)
            rr[x + "_D"] = np.where(keep[ti], Dx[:, LATE].mean(1), np.nan)
            both = d["own"][:, -1] & ref["own"][ti, -1]
            dI[x] = np.where(both, Im(d)[:, -1] - Iref[ti, -1], np.nan)
            sv[x] = d["own"][keep[ti], -1].mean()
            curves.setdefault((v, x), []).append(np.nanmean((Dx / np.where(okD[ti], D0[ti], np.nan)[:, None])[keep[ti]], 0))
        tw = c["twin"]; tw2 = c["twin2"]
        twE = np.abs(Em(tw) - Eref); curves.setdefault((v, "twin"), []).append(np.nanmean(twE / np.where(okD, D0, np.nan)[:, None], 0))
        acc["r_cl"].append(np.nanmean(rr["cl"])); acc["r_clamp"].append(np.nanmean(rr["clamp"])); acc["r_blind"].append(np.nanmean(rr["blind"]))
        acc["A"].append(np.nanmean(rr["cl"] - rr["clamp"])); acc["B"].append(np.nanmean(rr["cl"] - rr["blind"]))
        acc["dI_cl"].append(np.nanmean(dI["cl"])); acc["dI_clamp"].append(np.nanmean(dI["clamp"])); acc["dI_blind"].append(np.nanmean(dI["blind"]))
        acc["AI"].append(np.nanmean(dI["cl"] - dI["clamp"]))
        acc["surv_cl"].append(sv["cl"]); acc["surv_clamp"].append(sv["clamp"]); acc["surv_blind"].append(sv["blind"]); acc["surv_ref"].append(ref["own"][:, -1].mean())
        acc["A_med"].append(np.nanmedian(rr["cl"] - rr["clamp"])); acc["A_win"].append(np.nanmean((rr["cl_D"] < rr["clamp_D"])[np.isfinite(rr["cl_D"])]))
        acc["n_keep"].append(int(keep.sum()))
        # hidden kicks scored on regulated variables vs the world-chaos floor (twin with a new weather key)
        def dev(d, key, win):
            ti = d.get("ti", np.arange(K)); both = d["own"] & ref["own"][ti]
            a = np.where(both, d[key], np.nan); b = np.where(both, ref[key][ti], np.nan)
            return np.nanmean(np.abs(a - b)[:, win], 1)
        for nm, kk in (("hk0.1", "hk01"), ("hk0.3", "hk03")):
            acc[f"{kk}_E"].append(np.nanmean(dev(c[nm], "E", LATE)))
            acc[f"{kk}_vs_tw"].append(np.nanmean(np.log(dev(c[nm], "E", LATE) + 1e-6) - np.log(dev(tw, "E", LATE) + 1e-6)))
        acc["tw_E"].append(np.nanmean(dev(tw, "E", LATE))); acc["tw_E_early"].append(np.nanmean(dev(tw, "E", EARLY))); acc["hk01_E_early"].append(np.nanmean(dev(c["hk0.1"], "E", EARLY)))
        acc["hk01_I"].append(np.nanmean(dev(c["hk0.1"], "c_int", LATE))); acc["tw_I"].append(np.nanmean(dev(tw, "c_int", LATE)))
        acc["hk01_PE"].append(np.nanmean(dev(c["hk0.1"], "pestep", LATE))); acc["tw_PE"].append(np.nanmean(dev(tw, "pestep", LATE)))
        for nm in ("hk0.1", "twin"):
            d = c[nm]; ti = d.get("ti", np.arange(K)); both = d["own"] & ref["own"][ti]
            curves.setdefault((v, "dE_" + nm), []).append(np.nanmean(np.where(both, np.abs(d["E"] - ref["E"][ti]), np.nan), 0))
    if not acc["A"]: continue
    rep["runs"][v] = {k: [float(x) for x in acc[k]] for k in acc}
    rep["ed"][v] = {k: ci(acc[k]) for k in ("r_cl", "r_clamp", "r_blind", "A", "B", "dI_cl", "dI_clamp", "dI_blind", "AI", "surv_cl", "surv_clamp", "surv_blind", "surv_ref", "A_med", "A_win", "n_keep")}
    rep["hk"][v] = {k: ci(acc[k]) for k in ("hk01_E", "hk03_E", "tw_E", "hk01_vs_tw", "hk03_vs_tw", "hk01_I", "tw_I", "hk01_PE", "tw_PE", "tw_E_early", "hk01_E_early")}
    A, B = rep["ed"][v]["A"], rep["ed"][v]["B"]
    W = rep["ed"][v]["A_win"]
    act = np.isfinite(A[1]) and A[1] > 0 and np.isfinite(W[1]) and W[1] > 0.5     # mean A > 0 AND most targets recover better closed-loop
    rep["verdict"][v] = dict(seeds=A[3], energy_recovery="active > passive" if act else ("passive ≥ active" if np.isfinite(A[2]) and A[2] < 0 else "no detectable difference"),
                             via_sensing=bool(np.isfinite(B[1]) and B[1] > 0), kicks_beyond_world_noise=bool(rep["hk"][v]["hk01_vs_tw"][1] > 0))
L += ["## Energy-drop test (50% of energy removed at t=0; recovery fraction r = 1 − mean deficit over steps 60–120 / amount removed)\n",
      "| variant | seeds | r closed | r clamped (passive) | r sensor-blind | A = r_cl − r_clamp | B = r_cl − r_blind | survival at 120 (cl / clamp / blind, targets whose reference survives; last = reference survival over all targets) |", "|---|---|---|---|---|---|---|---|"]
for v, e in rep["ed"].items():
    L.append(f"| {v} | {e['A'][3]} | {f(e['r_cl'])} | {f(e['r_clamp'])} | {f(e['r_blind'])} | {f(e['A'])} | {f(e['B'])} | {e['surv_cl'][0]:.2f} / {e['surv_clamp'][0]:.2f} / {e['surv_blind'][0]:.2f} / {e['surv_ref'][0]:.2f} |")
L += ["\n## Robustness: median of per-target A, and fraction of targets where the closed loop ends with the smaller deficit (0.5 = chance)\n", "| variant | targets used per seed | median A | P(closed deficit < clamped deficit) |", "|---|---|---|---|"]
for v, e in rep["ed"].items():
    L.append(f"| {v} | {','.join(str(int(x)) for x in rep['runs'][v]['n_keep'])} | {f(e['A_med'])} | {f(e['A_win'])} |")
L += ["\n## Extra cumulative intake by step 120 relative to the unperturbed reference (agents alive in both)\n", "| variant | closed | clamped | blind | closed − clamped |", "|---|---|---|---|---|"]
for v, e in rep["ed"].items():
    L.append(f"| {v} | {f(e['dI_cl'])} | {f(e['dI_clamp'])} | {f(e['dI_blind'])} | {f(e['AI'])} |")
L += ["\n## Hidden-state kicks scored on regulated variables vs world-chaos floor (mean |Δ| over steps 60–120 vs unperturbed reference)\n",
      "| variant | |ΔE| kick 0.1 | |ΔE| kick 0.3 | |ΔE| twin (new weather) | log(kick0.1/twin) | log(kick0.3/twin) | |ΔE| early (0–20) kick0.1 / twin | |Δintake| kick0.1 / twin | |ΔPE| kick0.1 / twin |", "|---|---|---|---|---|---|---|---|---|"]
for v, h in rep["hk"].items():
    L.append(f"| {v} | {f(h['hk01_E'])} | {f(h['hk03_E'])} | {f(h['tw_E'])} | {f(h['hk01_vs_tw'])} | {f(h['hk03_vs_tw'])} | {h['hk01_E_early'][0]:.3f} / {h['tw_E_early'][0]:.3f} | {h['hk01_I'][0]:.3f} / {h['tw_I'][0]:.3f} | {h['hk01_PE'][0]:.4f} / {h['tw_PE'][0]:.4f} |")
L += ["\n## Verdict (active recovery requires the 95% CI of mean A to exclude 0 on the positive side AND the CI of P(closed deficit < clamped deficit) to exclude 0.5)\n", "| variant | energy recovery | via sensing (B > 0) | kick effect on E exceeds world noise |", "|---|---|---|---|"]
for v, d in rep["verdict"].items(): L.append(f"| {v} | {d['energy_recovery']} | {d['via_sensing']} | {d['kicks_beyond_world_noise']} |")
L += ["\n## Validation\n", f"Clamping an unperturbed target to its own reference actions reproduces the reference energy exactly while the reference agent lives (targets whose reference survives 120 steps; max |ΔE| = {np.nanmax([x['clamp0_maxdiff'] for x in rep['validation'].values()]):.2e}). Over all targets it is {max(x['clamp0_maxdiff_all'] for x in rep['validation'].values()):.2e}, because the recorded action is zeroed on the step the reference agent dies; that is why energy-drop metrics use only targets whose reference survives.",
      f"Targets used per run (reference alive at step 120): {sum(x['n_keep'] for x in rep['validation'].values())} of {sum(x['K'] for x in rep['validation'].values())}."]
open(os.path.join(O, "tables6.md"), "w").write("\n".join(L) + "\n")
json.dump(rep, open(os.path.join(O, "report6.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
vs = [v for v in V if (v, "cl") in curves]
fig, ax = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3.6), sharey=True); ax = np.atleast_1d(ax)
for a, v in zip(ax, vs):
    for x, cc in (("cl", "tab:red"), ("clamp", "tab:blue"), ("blind", "tab:green"), ("twin", "0.5")):
        for i, s in enumerate(curves[(v, x)]): a.plot(s, color=cc, lw=1, alpha=.8, ls="--" if x == "twin" else "-", label=({"cl": "closed", "clamp": "clamped (passive)", "blind": "sensor-blind", "twin": "|ΔE| twin floor"}[x] if i == 0 else None))
    a.axhline(0, color="0.7", lw=.5); a.set_title(v, fontsize=9); a.set_xlabel("step after 50% energy drop")
ax[0].set_ylabel("deficit / amount removed"); ax[0].legend(fontsize=6); fig.suptitle("Exp6 energy-drop recovery (lines = seeds; lower = more recovered)"); fig.tight_layout(); fig.savefig(os.path.join(O, "exp6_energy_drop.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (key, t) in enumerate((("A", "A = r_closed − r_clamped (> 0 = active)"), ("B", "B = r_closed − r_sensor-blind (> 0 = via sensing)"))):
    for i, v in enumerate(vs):
        m, lo, hi, _ = rep["ed"][v][key]; ax[j].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[v], capsize=4)
        ax[j].scatter([i] * len(rep["runs"][v][key]), rep["runs"][v][key], color=COL[v], s=10, alpha=.5)
    ax[j].axhline(0, color="0.5", lw=.7); ax[j].set_xticks(range(len(vs))); ax[j].set_xticklabels(vs, rotation=30, fontsize=8); ax[j].set_title(t)
fig.suptitle("Exp6 energy-drop contrasts (95% t-CI over 3 seeds)"); fig.tight_layout(); fig.savefig(os.path.join(O, "exp6_contrasts.png"), dpi=110); plt.close(fig)
fig, ax = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3.6), sharey=True); ax = np.atleast_1d(ax)
for a, v in zip(ax, vs):
    for nm, cc, lab in (("dE_hk0.1", "tab:red", "hidden kick 0.1"), ("dE_twin", "0.4", "twin (new weather)")):
        for i, s in enumerate(curves[(v, nm)]): a.plot(s, color=cc, lw=1, label=lab if i == 0 else None)
    a.set_title(v, fontsize=9); a.set_xlabel("step")
ax[0].set_ylabel("mean |E − E_ref|"); ax[0].legend(fontsize=7); fig.suptitle("Exp6 hidden kicks scored on energy vs world-chaos floor"); fig.tight_layout(); fig.savefig(os.path.join(O, "exp6_kicks_energy.png"), dpi=110); plt.close(fig)
print(open(os.path.join(O, "tables6.md")).read())
