"""Analysis for Experiment 14 (perturbation recovery on the Exp 13 t=10000 / t=5000 forks). Reads runs14/p14A_*, p14B_*,
replaycheck.json. Writes out14/: tables14.md, report14.json, exp14_recovery.png, exp14_primary.png, exp14_energy.png.
Per run (arm x seed x horizon) metrics are averaged over targets; 95% t-CIs over seeds (n=9, df=8), paired by seed."""
import os, glob, pickle, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs14"); O = os.path.join(HERE, "out14"); os.makedirs(O, exist_ok=True)
V = ["sw_full", "sw_mg2"]; TS = [10000, 5000]; SEEDS = list(range(9))
CONDS = ["kick0.03", "kick0.1", "kick0.3", "lyap", "ongoing"]
LATE = slice(60, 120)
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else np.nan, np.nan, np.nan, n, np.nan)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    return (float(m), float(m - h), float(m + h), n, p)
f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]" if np.isfinite(c[0]) else "n/a"
fp = lambda c: f"{f(c)}, p = {c[4]:.3g}" if np.isfinite(c[4]) else f(c)
rep = dict(A={}, B={}, primary={}, contrasts={}, validation={}, n_runs={})
# ---------------- phase A: Exp 5 protocol ----------------
A = {}; curves = {}
for p in sorted(glob.glob(os.path.join(R, "p14A_*.pkl"))):
    d = pickle.load(open(p, "rb")); A[(d["variant"], d["tf"], d["seed"])] = d
def phaseA_seed(d, c):
    r = d["conds"][c]; ok = r["alive_cl"][:, 50] & r["alive_ref"][:, 50]
    d0 = np.full(len(ok), 0.01 * np.sqrt(16)) if c == "ongoing" else r["d0"]
    good = ok & np.isfinite(d0) & (d0 > 0)
    if good.sum() == 0: return None
    lr = lambda x: np.log(np.maximum(x[:, 50], 1e-12) / d0)
    cl, ol, fr = lr(r["d_cl"])[good], lr(r["d_ol"])[good], lr(r["d_fr"])[good]
    out = dict(cl=cl.mean(), ol=ol.mean(), fr=fr.mean(), dco=(cl - ol).mean(), dcf=(cl - fr).mean(), n=int(good.sum()),
               win_ol=float(np.mean(cl < ol)), win_fr=float(np.mean(cl < fr)))
    okl = r["alive_cl"][:, 99] & r["alive_ref"][:, 99]
    out["og"] = (np.log(r["d_cl"][okl, 50:100].mean(1) + 1e-12).mean() - np.log(r["d_ol"][okl, 50:100].mean(1) + 1e-12).mean()) if (c == "ongoing" and okl.sum()) else np.nan
    out["xc"] = (r["cost_cl"].sum(1) - r["cost_ref"].sum(1))[okl].mean() if okl.sum() else np.nan
    out["dE"] = (r["E_cl"][:, 99] - r["E_ref"][:, 99])[okl].mean() if okl.sum() else np.nan
    with np.errstate(divide="ignore"):
        out["curve"] = np.stack([np.nanmean(np.log(np.maximum(r[k][good], 1e-12) / d0[good, None]), 0) for k in ("d_cl", "d_ol", "d_fr")])
    return out
PS = {}
for v in V:
    for tf in TS:
        for c in CONDS:
            per = [phaseA_seed(A[(v, tf, s)], c) if (v, tf, s) in A else None for s in SEEDS]
            PS[(v, tf, c)] = per
            got = [x for x in per if x is not None]
            if not got: continue
            arr = {k: np.array([x[k] if x is not None else np.nan for x in per]) for k in ("cl", "ol", "fr", "dco", "dcf", "og", "xc", "dE", "win_ol", "win_fr")}
            rec_co = -arr["dco"]; rec_cf = -arr["dcf"]
            row = dict(seeds=len(got), n=int(sum(x["n"] for x in got)), cl=ci(arr["cl"]), ol=ci(arr["ol"]), fr=ci(arr["fr"]),
                       rec_co=ci(rec_co), rec_cf=ci(rec_cf), og=ci(arr["og"]), xc=ci(arr["xc"]), dE=ci(arr["dE"]),
                       win_ol=ci(arr["win_ol"]), win_fr=ci(arr["win_fr"]), per_seed={k: [float(x) for x in arr[k]] for k in arr})
            row["per_seed"]["rec_co"] = [float(x) for x in rec_co]
            rep["A"].setdefault(f"{v}@{tf}", {})[c] = row
            curves[(v, tf, c)] = np.stack([x["curve"] for x in got])
# primary
pr = rep["A"].get("sw_full@10000", {}).get("kick0.1")
if pr:
    rep["primary"] = dict(definition="sw_full, t=10000, kick eps=0.1: recovery_closed - recovery_open = -(log r_cl(50) - log r_ol(50)), per-seed mean over targets alive in both runs at step 50; paired n=9; 95% t CI > 0 required",
                          estimate=pr["rec_co"], per_seed=pr["per_seed"]["rec_co"], n_seeds=pr["seeds"], n_targets=pr["n"],
                          holds=bool(np.isfinite(pr["rec_co"][1]) and pr["rec_co"][1] > 0 and pr["seeds"] == 9),
                          seeds_positive=int(np.sum(np.array(pr["per_seed"]["rec_co"]) > 0)))
# paired contrasts (secondary)
def paired(key_a, key_b, c, k="rec_co"):
    a = rep["A"].get(key_a, {}).get(c); b = rep["A"].get(key_b, {}).get(c)
    if not (a and b): return None
    return ci(np.array(a["per_seed"][k]) - np.array(b["per_seed"][k]))
for c in ("kick0.1", "ongoing"):
    k = "rec_co" if c != "ongoing" else "og"
    for lab, ka, kb in (("t10000 − t5000, sw_full", "sw_full@10000", "sw_full@5000"), ("t10000 − t5000, sw_mg2", "sw_mg2@10000", "sw_mg2@5000"),
                        ("sw_mg2 − sw_full, t10000", "sw_mg2@10000", "sw_full@10000"), ("sw_mg2 − sw_full, t5000", "sw_mg2@5000", "sw_full@5000")):
        r_ = paired(ka, kb, c, k)
        if r_: rep["contrasts"][f"{lab} | {c} | {'closed−open recovery' if k == 'rec_co' else 'ongoing log(cl/ol)'}"] = r_
# ---------------- phase B: Exp 6 energy-drop ----------------
B = {}
for p in sorted(glob.glob(os.path.join(R, "p14B_*.pkl"))):
    d = pickle.load(open(p, "rb")); B[(d["variant"], d["tf"], d["seed"])] = d
ecurves = {}
def phaseB_seed(r):
    c = r["conds"]; ref = c["ref"]; K = len(r["targets"])
    keep = ref["own"][:, -1]
    Em = lambda d: np.where(d["own"], d["E"], 0.0)
    Im = lambda d: np.where(d["own"], d["c_int"], np.nan)
    Eref = Em(ref); Iref = Im(ref); D0 = r["E0"] * r["drop"]; okD = D0 > 1e-3
    rr, dI, sv = {}, {}, {}
    for x in ("cl", "clamp", "blind"):
        d = c[f"ed_{x}"]; ti = d["ti"]
        Dx = Eref[ti] - Em(d)
        rr[x] = np.where(keep[ti], 1 - Dx[:, LATE].mean(1) / np.where(okD[ti], D0[ti], np.nan), np.nan)
        rr[x + "_D"] = np.where(keep[ti], Dx[:, LATE].mean(1), np.nan)
        both = d["own"][:, -1] & ref["own"][ti, -1]
        dI[x] = np.where(both, Im(d)[:, -1] - Iref[ti, -1], np.nan)
        sv[x] = d["own"][keep[ti], -1].mean() if keep[ti].any() else np.nan
        ecurves.setdefault((r["variant"], r["tf"], x), []).append(np.nanmean((Dx / np.where(okD[ti], D0[ti], np.nan)[:, None])[keep[ti]], 0))
    def dev(d, key, win):
        ti = d.get("ti", np.arange(K)); both = d["own"] & ref["own"][ti]
        a = np.where(both, d[key], np.nan); b = np.where(both, ref[key][ti], np.nan)
        return np.nanmean(np.abs(a - b)[:, win], 1)
    tw = c["twin"]; hk = c["hk0.1"]
    v0 = c["clamp0"]; kv = keep[v0["ti"]]
    with np.errstate(all="ignore"):
        out = dict(r_cl=np.nanmean(rr["cl"]), r_clamp=np.nanmean(rr["clamp"]), r_blind=np.nanmean(rr["blind"]),
                   A=np.nanmean(rr["cl"] - rr["clamp"]), Bv=np.nanmean(rr["cl"] - rr["blind"]),
                   A_med=np.nanmedian(rr["cl"] - rr["clamp"]), A_win=np.nanmean((rr["cl_D"] < rr["clamp_D"])[np.isfinite(rr["cl_D"])]),
                   AI=np.nanmean(dI["cl"] - dI["clamp"]), surv_cl=sv["cl"], surv_clamp=sv["clamp"], surv_blind=sv["blind"], surv_ref=ref["own"][:, -1].mean(),
                   n_keep=int(keep.sum()), hk_E=np.nanmean(dev(hk, "E", LATE)), tw_E=np.nanmean(dev(tw, "E", LATE)),
                   hk_vs_tw=np.nanmean(np.log(dev(hk, "E", LATE) + 1e-6) - np.log(dev(tw, "E", LATE) + 1e-6)),
                   clamp0=float(np.abs(v0["E"][kv] - ref["E"][v0["ti"][kv]]).max()) if kv.any() else np.nan)
    return out
EK = ("r_cl", "r_clamp", "r_blind", "A", "Bv", "A_med", "A_win", "AI", "surv_cl", "surv_clamp", "surv_blind", "surv_ref", "n_keep", "hk_E", "tw_E", "hk_vs_tw")
for v in V:
    for tf in TS:
        per = [phaseB_seed(B[(v, tf, s)]) if (v, tf, s) in B else None for s in SEEDS]
        got = [x for x in per if x is not None]
        if not got: continue
        arr = {k: np.array([x[k] if x is not None else np.nan for x in per], float) for k in EK + ("clamp0",)}
        row = {k: ci(arr[k]) for k in EK}; row["seeds"] = len(got); row["per_seed"] = {k: [float(x) for x in arr[k]] for k in EK}
        Aci, W = row["A"], row["A_win"]
        row["verdict"] = "active > passive" if (np.isfinite(Aci[1]) and Aci[1] > 0 and np.isfinite(W[1]) and W[1] > 0.5) else ("passive ≥ active" if np.isfinite(Aci[2]) and Aci[2] < 0 else "no detectable difference")
        rep["B"][f"{v}@{tf}"] = row
for v in V:
    a = rep["B"].get(f"{v}@10000"); b = rep["B"].get(f"{v}@5000")
    if a and b: rep["contrasts"][f"t10000 − t5000, {v} | energy drop | A = r_cl − r_clamp"] = ci(np.array(a["per_seed"]["A"]) - np.array(b["per_seed"]["A"]))
for tf in TS:
    a = rep["B"].get(f"sw_mg2@{tf}"); b = rep["B"].get(f"sw_full@{tf}")
    if a and b: rep["contrasts"][f"sw_mg2 − sw_full, t{tf} | energy drop | A = r_cl − r_clamp"] = ci(np.array(a["per_seed"]["A"]) - np.array(b["per_seed"]["A"]))
# ---------------- validation ----------------
val = rep["validation"]
val["fork_load_exact_all"] = all(d["fork_check"]["exact"] and d["fork_check"]["shape_ok"] and not d["fork_check"]["missing"] for d in list(A.values()) + list(B.values()))
val["fork_t_matches"] = all(d["fork_check"]["t"] == d["tf"] for d in list(A.values()) + list(B.values()))
val["match_err_max"] = max((d["match_err"] for d in A.values()), default=np.nan)
val["ref_repeat_equal_all"] = all(d["ref_repeat_equal"] for d in A.values())
c0 = [phaseB_seed(d)["clamp0"] for d in B.values()]
val["clamp0_maxdiff"] = float(np.nanmax(c0)) if c0 else np.nan
val["clamp0_maxdiff_all_targets"] = max((d["clamp0_maxdiff"] for d in B.values()), default=np.nan)
# phase-A ref (sim9, ctl=None) vs phase-B ref (sim9c, inactive hook): same physics, different XLA compilation
dd = []
for k in A:
    if k in B:
        a = A[k]; b = B[k]["conds"]["ref"]
        al = (a["ref_alive"].T & b["alive"][:, :100])
        dd.append(dict(E=float(np.abs(b["E"][:, :100] - a["ref_E"].T)[al].max()) if al.any() else 0.0,
                       first_diff_step=int(np.argmax(np.any(np.abs(b["E"][:, :100] - a["ref_E"].T) > 0, 0))) if np.any(b["E"][:, :100] != a["ref_E"].T) else -1,
                       E10=float(np.abs(b["E"][:, :10] - a["ref_E"].T[:, :10]).max())))
val["refA_vs_refB"] = dict(n=len(dd), max_absdiff_E_100=max((x["E"] for x in dd), default=np.nan), max_absdiff_E_first10=max((x["E10"] for x in dd), default=np.nan),
                           identical_runs=int(sum(x["first_diff_step"] < 0 for x in dd)))
rc = os.path.join(R, "replaycheck.json")
val["replaycheck"] = json.load(open(rc)) if os.path.exists(rc) else None
val["n_targets_per_run_A"] = {f"{k[0]}@{k[1]}_s{k[2]}": int(len(d["targets"])) for k, d in sorted(A.items())}
rep["n_runs"] = dict(A=len(A), B=len(B))
# ---------------- tables ----------------
L = ["# Experiment 14: perturbation recovery on long-evolved populations (Exp 13 forks) — tables",
     "Per-run means over targets; 95% t-CI over seeds (n = 9 → df = 8), paired by seed. Recovery = −log(d(50)/ε); positive closed − open = closed loop recovers more.\n",
     "## Pre-registered primary (fixed 2026-11-13; operationalised 2026-11-14 before data)\n"]
if rep["primary"]:
    P = rep["primary"]
    L += [f"- {P['definition']}", f"- **closed − open-loop recovery = {fp(P['estimate'])}**, n = {P['n_seeds']} seeds ({P['n_targets']} targets), seeds positive {P['seeds_positive']}/9 → **{'HOLDS' if P['holds'] else 'DOES NOT HOLD'}**",
          "- per seed: " + ", ".join(f"{x:+.2f}" for x in P["per_seed"])]
L += ["\n## Secondary: Exp 5 protocol, all arms × horizons × conditions\n",
      "| arm @ t | cond | seeds | targets | log r_cl(50) | log r_ol(50) | log r_fr(50) | closed − open recovery | closed − frozen recovery | P(cl < ol) | ongoing log(cl/ol) 50–100 | extra cost (100 steps) | ΔE(100) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key, conds in rep["A"].items():
    for c, r in conds.items():
        L.append(f"| {key} | {c} | {r['seeds']} | {r['n']} | {f(r['cl'])} | {f(r['ol'])} | {f(r['fr'])} | {f(r['rec_co'])} | {f(r['rec_cf'])} | {f(r['win_ol'])} | {f(r['og']) if c == 'ongoing' else ''} | {f(r['xc'])} | {f(r['dE'])} |")
L += ["\n## Secondary: Exp 6 energy-drop test (50% of energy removed at t = 0; r = 1 − mean deficit over steps 60–120 / amount removed; targets whose reference survives 120 steps)\n",
      "| arm @ t | seeds | targets kept/seed (mean) | r closed | r clamped (passive) | r sensor-blind | A = r_cl − r_clamp | B = r_cl − r_blind | median A | P(closed deficit < clamped) | survival cl / clamp / blind / ref | verdict |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for key, r in rep["B"].items():
    L.append(f"| {key} | {r['seeds']} | {r['n_keep'][0]:.1f} | {f(r['r_cl'])} | {f(r['r_clamp'])} | {f(r['r_blind'])} | {f(r['A'])} | {f(r['Bv'])} | {f(r['A_med'])} | {f(r['A_win'])} | "
             f"{r['surv_cl'][0]:.2f} / {r['surv_clamp'][0]:.2f} / {r['surv_blind'][0]:.2f} / {r['surv_ref'][0]:.2f} | {r['verdict']} |")
L += ["\n## Secondary: hidden kick ε = 0.1 scored on energy vs world-chaos floor (twin with new weather key), |ΔE| mean over steps 60–120\n",
      "| arm @ t | |ΔE| kick 0.1 | |ΔE| twin | log(kick/twin) |", "|---|---|---|---|"]
for key, r in rep["B"].items():
    L.append(f"| {key} | {f(r['hk_E'])} | {f(r['tw_E'])} | {f(r['hk_vs_tw'])} |")
L += ["\n## Secondary paired contrasts (post hoc / not pre-registered)\n", "| contrast | measure | estimate [95% CI], p |", "|---|---|---|"]
for k, r in rep["contrasts"].items():
    lab, c, m = k.split(" | "); L.append(f"| {lab} ({c}) | {m} | {fp(r)} |")
L += ["\n## Validation\n",
      f"- Fork loads: every sim9 state field present, shape-matched and equal to the saved npz arrays: {val['fork_load_exact_all']}; fork step counter equals horizon: {val['fork_t_matches']}.",
      f"- Exp 13 harness stepped from t = 5000 to 10000 (sw_full, seeds 0–2) vs saved t = 10000 forks (all fields except host-managed `anc`): " + (f"bit-for-bit = {val['replaycheck']['bit_for_bit']}" + ("" if val['replaycheck']['bit_for_bit'] else f", mismatched {[v['mismatched'] for v in val['replaycheck']['per_seed'].values()]}") if val["replaycheck"] else "not available"),
      f"- Unperturbed reference recomputed in a second, differently composed batch: bit-identical in all phase-A runs = {val['ref_repeat_equal_all']}.",
      f"- Host-side open-loop recursion vs simulated hidden state (first 10 steps, float64 vs float32): max |Δh| = {val['match_err_max']:.1e}.",
      f"- Clamp validation (no drop, actions clamped to the reference's own actions; targets whose reference survives): max |ΔE| = {val['clamp0_maxdiff']:.1e} (all clamp0 targets incl. those whose reference dies: {val['clamp0_maxdiff_all_targets']:.1e}).",
      f"- Phase-A reference (sim9, ctl=None) vs phase-B reference (sim9c, inactive hook): same physics, different XLA compilation; max |ΔE| over first 10 steps = {val['refA_vs_refB']['max_absdiff_E_first10']:.1e}, over 100 steps = {val['refA_vs_refB']['max_absdiff_E_100']:.1e} ({val['refA_vs_refB']['identical_runs']}/{val['refA_vs_refB']['n']} runs identical). Every contrast compares conditions within one phase (one compiled function), so this float-reassociation drift does not enter any contrast.",
      f"- Runs: phase A {rep['n_runs']['A']}/36, phase B {rep['n_runs']['B']}/36."]
open(os.path.join(O, "tables14.md"), "w").write("\n".join(L) + "\n")
json.dump(rep, open(os.path.join(O, "report14.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if o is None else str(o)))
# ---------------- figures ----------------
if curves:
    keys = [(v, tf) for v in V for tf in TS if (v, tf, "kick0.1") in curves]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6), sharey=True); ax = np.atleast_1d(ax)
    for a, (v, tf) in zip(ax, keys):
        cv = curves[(v, tf, "kick0.1")]
        for i, (nm, cc) in enumerate((("closed", "tab:red"), ("open-loop", "tab:blue"), ("frozen", "tab:green"))):
            for j, s in enumerate(cv): a.plot(s[i], color=cc, lw=1, alpha=.6, label=nm if j == 0 else None)
        a.axhline(0, color="0.6", lw=.5); a.axvline(50, color="0.8", lw=.5); a.set_title(f"{v} @ t={tf}"); a.set_xlabel("step after kick (ε=0.1)")
    ax[0].set_ylabel("mean log d(t)/ε"); ax[0].legend(fontsize=7); fig.suptitle("Exp 14 perturbation recovery on Exp 13 forks (lines = seeds)"); fig.tight_layout()
    fig.savefig(os.path.join(O, "exp14_recovery.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (c, k, t) in enumerate((("kick0.1", "rec_co", "closed − open-loop recovery @50 (kick 0.1)"), ("ongoing", "og", "ongoing: log(closed/open) t50–100 (<0 = closed better)"))):
        for i, key in enumerate([f"{v}@{tf}" for v in V for tf in TS]):
            r = rep["A"].get(key, {}).get(c)
            if not r: continue
            m, lo, hi = r[k][:3]; ax[j].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k" if key == "sw_full@10000" else "0.5", capsize=4)
            ax[j].scatter([i] * 9, r["per_seed"][k], s=10, alpha=.5)
        ax[j].axhline(0, color="0.5", lw=.7); ax[j].set_xticks(range(4)); ax[j].set_xticklabels([f"{v}@{tf}" for v in V for tf in TS], rotation=20); ax[j].set_title(t, fontsize=9)
    fig.suptitle("Black = pre-registered primary (sw_full @ t=10000); 95% t-CI, n=9"); fig.tight_layout(); fig.savefig(os.path.join(O, "exp14_primary.png"), dpi=110); plt.close(fig)
if ecurves:
    keys = [(v, tf) for v in V for tf in TS if (v, tf, "cl") in ecurves]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.6), sharey=True); ax = np.atleast_1d(ax)
    for a, (v, tf) in zip(ax, keys):
        for x, cc in (("cl", "tab:red"), ("clamp", "tab:blue"), ("blind", "tab:green")):
            m = np.nanmean(np.stack(ecurves[(v, tf, x)]), 0); a.plot(m, color=cc, label=x)
        a.axhline(0, color="0.6", lw=.5); a.set_title(f"{v} @ t={tf}"); a.set_xlabel("step after 50% energy drop")
    ax[0].set_ylabel("energy deficit / amount removed (mean over seeds)"); ax[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(O, "exp14_energy.png"), dpi=110); plt.close(fig)
print(open(os.path.join(O, "tables14.md")).read())
