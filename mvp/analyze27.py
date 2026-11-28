"""Exp 27 analysis (pre-registered in NIGHTLY_LOG 2026-11-28). Assay validation on constructed systems with known energy feedback.
Per system and replicate (9 seeds): A_surv = S_CL - S_OPEN, B_surv = S_CL - S_FROZEN (survival at t=150), 95% t CI; detected iff validity
and both lower bounds > 0. Sensitivity = detection rate over 5 replicates; FPR over 6 blind controls x 5 replicates; pooled = per-seed mean
over replicates, n=9. Writes figures/exp27/{tables27.md, report27.json, exp27_dose_response.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp27_run as X
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp27"); os.makedirs(O, exist_ok=True); R = os.path.join(HERE, "runs27")
def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
f3 = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
D = {}
for f in glob.glob(os.path.join(R, "a27_*.pkl")):
    d = pickle.load(open(f, "rb")); D.setdefault((d["name"] + ("@sighted" if d.get("sighted") else "") + ("@x26" if "_x26" in f else "") + ("@lead" if "_lead" in f else ""), d["haz"]), {}).setdefault(d["rep"], {})[d["seed"]] = d
sv = lambda d, a: float(d["arms"][a]["alive"][149].mean())
ORDER = ["P0", "P0.25", "P0.5", "P1", "P2", "P4", "P8", "P16", "STEP", "ES", "CONST0.8", "SHAM_STEP", "SHAM_P4", "ES_BLIND", "ES_ABL"]
NOMG = {"P0": 0, **{f"P{g}": g for g in X.GAINS}}
rep = dict(systems={}); L = ["# Exp 27 tables: assay validation on constructed systems with known energy feedback (survival at t = 150, K = 200 targets/seed, 50% drop)\n",
    "All systems are constructed controls (hand-coded or trained in this sim); brains' energy sense clamped (blind = 1), so the controller is the only energy→action route.\n"]
def summarise(name, haz):
    R_ = D.get((name, haz), {}); out = dict(reps={}, complete=True)
    reps = sorted(r for r in R_ if len(R_[r]) == 9)
    if not reps: return None
    per_seed_A = {s: [] for s in range(9)}; per_seed_B = {s: [] for s in range(9)}; per_seed_D = {s: [] for s in range(9)}; ge = []; valid_all = True
    for r in reps:
        ss = sorted(R_[r]); dd = R_[r]
        valid = all(dd[s]["clamp0_exact"] and dd[s]["births"] == 0 for s in ss) and (all(dd[s]["frozen_eq_cl"] for s in ss) if name in X.IGNORES_INPUT else True)
        valid_all &= valid
        A = [sv(dd[s], "cl") - sv(dd[s], "open") for s in ss]; B = [sv(dd[s], "cl") - sv(dd[s], "frozen") for s in ss]; Dd = [sv(dd[s], "ref") - sv(dd[s], "cl") for s in ss]
        for s, a, b, x in zip(ss, A, B, Dd): per_seed_A[s].append(a); per_seed_B[s].append(b); per_seed_D[s].append(x)
        cA, cB = ci(A), ci(B)
        det = bool(valid and cA[1] > 0 and cB[1] > 0); detB = bool(valid and cB[1] > 0); detA = bool(valid and cA[1] > 0)
        ge += [dd[s]["G_eff"] for s in ss]
        out["reps"][r] = dict(A=cA, B=cB, valid=valid, detected=det, detB=detB, detA=detA, nposB=int(sum(b > 0 for b in B)))
    pA = ci([np.mean(per_seed_A[s]) for s in range(9)]); pB = ci([np.mean(per_seed_B[s]) for s in range(9)]); pD = ci([np.mean(per_seed_D[s]) for s in range(9)])
    surv = {a: float(np.mean([sv(R_[r][s], a) for r in reps for s in range(9)])) for a in ("ref", "cl", "open", "frozen")}
    out.update(n_reps=len(reps), valid=valid_all, det_rate=sum(out["reps"][r]["detected"] for r in reps) / len(reps), detB_rate=sum(out["reps"][r]["detB"] for r in reps) / len(reps),
               detA_rate=sum(out["reps"][r]["detA"] for r in reps) / len(reps), pooled_A=pA, pooled_B=pB, pooled_D=pD, G_eff=float(np.nanmean(ge)) if not all(np.isnan(ge)) else float("nan"), surv=surv)
    return out
H = "| system | ground truth | G_eff | reps | valid | detected (A&B) | B-only det. | pooled A_surv | pooled B_surv | cost of drop D | S ref / CL / OPEN / FROZEN |\n|---|---|---|---|---|---|---|---|---|---|---|"
def gt(n):
    n = n.split("@")[0]
    return "blind (neg. control)" if n in X.BLIND else ("trained, uses E?" if n == "ES" else "feedback (pos. control)")
for haz, title, ORD in ((0.004, "PRIMARY: predation world (haz = 0.004), R = 5 replicates × 9 seeds", ORDER), (0.0, "SECONDARY: no-predation world (haz = 0), R = 1", ORDER),
                        (0.004, "POST HOC diagnostics: brains' energy sense left ON (blind = 0, as in Exp 26), haz = 0.004. @sighted: R = 2 fresh draws; @x26: Exp 26 targets/key/hazard draws (R = 1); @lead: FROZEN fed ref POST-step E, the Exp 25/26 convention (R = 1)", ["STEP@sighted", "P1@sighted", "P0@sighted", "STEP@sighted@x26", "P0@sighted@x26", "STEP@sighted@x26@lead", "P0@sighted@x26@lead"])):
    L.append(f"\n## {title}\n\n" + H)
    for n in ORD:
        o = summarise(n, haz)
        if o is None: continue
        rep["systems"][f"{n}_h{haz:g}"] = o; s = o["surv"]
        L.append(f"| {n} | {gt(n)} | {o['G_eff']:.3f} | {o['n_reps']} | {o['valid']} | {o['det_rate']*o['n_reps']:.0f}/{o['n_reps']} | {o['detB_rate']*o['n_reps']:.0f}/{o['n_reps']} | {f3(o['pooled_A'])} | {f3(o['pooled_B'])} | {f3(o['pooled_D'])} | {s['ref']:.3f} / {s['cl']:.3f} / {s['open']:.3f} / {s['frozen']:.3f} |")
    L.append("\nPer-replicate B_surv (95% t CI, n = 9):\n")
    for n in ORD:
        o = rep["systems"].get(f"{n}_h{haz:g}")
        if o: L.append(f"- {n}: " + "; ".join(f"r{r} B {f3(v['B'])} A {f3(v['A'])}{' **DET**' if v['detected'] else ''}" for r, v in o["reps"].items()))
S = rep["systems"]; P = lambda n: S.get(f"{n}_h0.004")
# specificity
bl = [n for n in X.BLIND if P(n)]
fp = sum(sum(v["detected"] for v in P(n)["reps"].values()) for n in bl); nt = sum(len(P(n)["reps"]) for n in bl)
fpA = sum(sum(v["detA"] for v in P(n)["reps"].values()) for n in bl); fpB = sum(sum(v["detB"] for v in P(n)["reps"].values()) for n in bl)
rep["FPR"] = dict(controls=bl, tests=nt, detected=fp, A_only=fpA, B_only=fpB)
L.append(f"\n## Specificity (blind controls {bl})\n\n**FPR (pre-registered rule, A & B): {fp}/{nt}**; A-only lower bound > 0: {fpA}/{nt}; B-only: {fpB}/{nt}. "
         f"B false positives are impossible by construction for input-ignoring controllers ({[n for n in bl if n in X.IGNORES_INPUT]}: FROZEN == CL bit for bit, checked); the informative B tests are the shams.")
# g_min
lad = ["P0.25", "P0.5", "P1", "P2", "P4", "P8", "P16", "STEP"]
ok = {n: bool(P(n) and P(n)["det_rate"] >= 0.8 and P(n)["pooled_B"][1] > 0) for n in lad}
gmin = None
for i, n in enumerate(lad[:-1]):
    if all(ok[m] for m in lad[i:]): gmin = n; break
okB = {n: bool(P(n) and P(n)["detB_rate"] >= 0.8 and P(n)["pooled_B"][1] > 0) for n in lad}
gminB = next((n for i, n in enumerate(lad[:-1]) if all(okB[m] for m in lad[i:])), None)
rep.update(g_min=gmin, g_min_Bonly=gminB, ladder_ok=ok)
L.append(f"\n## Minimum detectable feedback\n\nPer-gain criterion (detection rate ≥ 4/5 and pooled B_surv lower bound > 0): {ok}. **g_min (pre-registered, A & B rule) = {gmin}**; with the B-only rule: {gminB}.")
# pay-check
PC = {}
for f in glob.glob(os.path.join(R, "p27_*.pkl")):
    d = pickle.load(open(f, "rb")); PC.setdefault((d["A"], d["B"]), {})[d["seed"]] = d
if PC:
    L.append("\n## Input-ablation pay-check (within-world parity, T = 300, keys PRNGKey(2500+seed), n = 9; survival)\n\n| comparison | S_A − S_B at t = 300 | at t = 150 | # > 0 |\n|---|---|---|---|")
    rep["pay"] = {}
    for (a, b), dd in sorted(PC.items()):
        ss = sorted(dd); d300 = [dd[s]["surv"][-1, 0] - dd[s]["surv"][-1, 1] for s in ss]; d150 = [dd[s]["surv"][149, 0] - dd[s]["surv"][149, 1] for s in ss]
        c3, c1 = ci(d300), ci(d150); rep["pay"][f"{a}-{b}"] = dict(t300=c3, t150=c1, n=len(ss), SA=float(np.mean([dd[s]['surv'][-1, 0] for s in ss])), SB=float(np.mean([dd[s]['surv'][-1, 1] for s in ss])))
        L.append(f"| S({a}) − S({b}) | {f3(c3)} | {f3(c1)} | {sum(x > 0 for x in d300)}/{len(ss)} |")
    u = rep["pay"].get("ES-ES_ABL"); rep["ES_uses_energy"] = bool(u and u["t300"][1] > 0)
    L.append(f"\n**Trained ES agent uses energy (U = S(ES) − S(ES_ABL) lower bound > 0 at T = 300): {rep.get('ES_uses_energy')}**")
for nm in ("es", "esblind"):
    p = os.path.join(R, f"es_{nm}.npz")
    if os.path.exists(p):
        z = np.load(p); rep[f"es_{nm}_hist"] = z["hist"].tolist(); rep[f"es_{nm}_theta"] = z["theta"].tolist()
open(os.path.join(O, "tables27.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report27.json"), "w"), indent=1, default=float)
# ---------- figure
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
gx = lambda n: P(n)["G_eff"]
ln = [n for n in ["P0.25", "P0.5", "P1", "P2", "P4", "P8", "P16", "STEP"] if P(n)]
for k, (key, lab) in enumerate((("pooled_B", "B_surv = S_CL − S_FROZEN"), ("pooled_A", "A_surv = S_CL − S_OPEN"))):
    a = ax[k]; xs = [gx(n) for n in ln]; m = [P(n)[key][0] for n in ln]; lo = [P(n)[key][1] for n in ln]; hi = [P(n)[key][2] for n in ln]
    a.errorbar(xs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt="o-", color="tab:blue", capsize=3, label="hand-coded servo ladder P(g) → STEP (pos. controls)")
    for n, x_, y_ in zip(ln, xs, m): a.annotate(n, (x_, y_), textcoords="offset points", xytext=(4, 6), fontsize=7)
    if P("ES"):
        o = P("ES"); a.errorbar([o["G_eff"]], [o[key][0]], yerr=[[o[key][0] - o[key][1]], [o[key][2] - o[key][0]]], fmt="D", color="tab:red", capsize=3, label="ES-trained agent")
    bx = -0.02
    for j, n in enumerate([n for n in X.BLIND if P(n)]):
        o = P(n); xx = 0.012 * (1 + j * 0.35)
        a.errorbar([xx], [o[key][0]], yerr=[[o[key][0] - o[key][1]], [o[key][2] - o[key][0]]], fmt="s", color="0.4", capsize=2, ms=4, label="energy-blind controls (G_eff = 0)" if j == 0 else None)
        a.annotate(n, (xx, o[key][2]), textcoords="offset points", xytext=(-6, 4), fontsize=6, rotation=70)
    h0 = [n for n in ["P0", "P1", "P4", "P16", "STEP"] if S.get(f"{n}_h0")]
    if h0: a.plot([max(S[f'{n}_h0']['G_eff'], 0.012) for n in h0], [S[f"{n}_h0"][key][0] for n in h0], "^--", color="tab:green", mfc="none", label="same ladder, no predation (secondary, R = 1)")
    if key == "pooled_B": a.axhspan(-0.012, 0.026, color="tab:orange", alpha=0.15, label="Exp 26 evolved pops' B_surv 95% CIs (span)")
    a.axhline(0, color="k", lw=0.6); a.set_xscale("log"); a.set_xlabel("drop-response gain G_eff (Δeat per unit energy lost; blind = 0, drawn at left)"); a.set_ylabel(lab + "  (pooled, n = 9 seeds, 95% CI)"); a.legend(fontsize=7, loc="upper left")
a = ax[2]; names = ["P0"] + ln + (["ES"] if P("ES") else []) + [n for n in X.BLIND if P(n) and n != "P0"]
dr = [P(n)["det_rate"] for n in names]; drB = [P(n)["detB_rate"] for n in names]; xx = np.arange(len(names))
a.bar(xx - 0.2, dr, 0.4, color=["0.5" if n in X.BLIND else ("tab:red" if n == "ES" else "tab:blue") for n in names], label="detected (A & B rule, pre-registered)")
a.bar(xx + 0.2, drB, 0.4, color="none", edgecolor="k", label="B-only rule")
a.axhline(0.8, ls=":", color="k", lw=0.8); a.set_xticks(xx); a.set_xticklabels(names, rotation=60, fontsize=8); a.set_ylabel("detection rate over 5 independent 9-seed replicates"); a.set_ylim(0, 1.05); a.legend(fontsize=7)
fig.suptitle(f"Exp 27: assay validation on constructed systems with known energy feedback (positive/negative controls, not evolved agents). g_min = {gmin}; FPR = {fp}/{nt}", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(O, "exp27_dose_response.png"), dpi=130); print("\n".join(L))
