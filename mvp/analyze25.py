"""Exp 25 analysis: starvation-predation pay-check (births off, survival currency). PRIMARY (haz = 0.004, T = 300, n seeds):
D* = S_COND - S_BLIND(e*), e* = argmax_e across-seed mean S_BLIND(e) (same seeds; conservative), and D_sham = S_COND - S_SHAM.
PASS iff mean D* >= +0.05 with 95% t CI lower bound > 0 AND D_sham CI lower bound > 0. Null control haz = 0 reported.
Writes figures/exp25/{tables25.md, report25.json, exp25_paycheck.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp25"); os.makedirs(O, exist_ok=True)
MARGIN = 0.05
def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
def per_seed(d, t):
    """S[(m,e)] = (S_COND, S_comp) at step index t, averaged over the two parities; plus secondaries."""
    o = {}
    for (m, e) in d["comps"]:
        sc, sx, pc, px, stc, stx, ec, ex, Ec, Ex = [], [], [], [], [], [], [], [], [], []
        for p in (0, 1):
            r = d["res"][(p, m, e)]; nA, nB = d["nA"][p], d["nB"][p]
            sc.append(r["alive"][t, 0] / nA); sx.append(r["alive"][t, 1] / nB)
            pc.append(r["pred"][:t + 1, 0].sum() / nA); px.append(r["pred"][:t + 1, 1].sum() / nB)
            stc.append(r["starve"][:t + 1, 0].sum() / nA); stx.append(r["starve"][:t + 1, 1].sum() / nB)
            ac = r["alive"][:t + 1, 0]; ax = r["alive"][:t + 1, 1]
            ec.append(r["eat"][:t + 1, 0].sum() / max(np.r_[nA, ac[:-1]].sum(), 1)); ex.append(r["eat"][:t + 1, 1].sum() / max(np.r_[nB, ax[:-1]].sum(), 1))
            Ec.append(r["Esum"][t, 0] / max(r["alive"][t, 0], 1)); Ex.append(r["Esum"][t, 1] / max(r["alive"][t, 1], 1))
        o[(m, e)] = dict(Sc=np.mean(sc), Sx=np.mean(sx), Pc=np.mean(pc), Px=np.mean(px), STc=np.mean(stc), STx=np.mean(stx), eatc=np.mean(ec), eatx=np.mean(ex), Ec=np.mean(Ec), Ex=np.mean(Ex))
    return o
D = {}
for f in sorted(glob.glob(os.path.join(HERE, "runs25", "run25_h*_s*.pkl"))):
    d = pickle.load(open(f, "rb")); D.setdefault(d["haz"], {})[d["seed"]] = d
rep = {}; L = ["# Exp 25 tables: starvation-predation pay-check (births off; currency = own survival; COND: eat = 1 iff own E < 0.40; within-world pairing, both uid parities averaged)\n"]
H = "| contrast | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"
def row(key, name, v):
    m, lo, hi, n = ci(v); rep[key] = dict(per_seed=[float(x) for x in v], mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v)))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[key]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[key]
figdat = {}
for haz in sorted(D, reverse=True):
    P = D[haz]; ss = sorted(P); comps = P[ss[0]]["comps"]; EG = [e for (m, e) in comps if m == "blind"]
    for t, tl in ((299, "T = 300"), (149, "t = 150")):
        S = {s: per_seed(P[s], t) for s in ss}
        mb = {e: float(np.mean([S[s][("blind", e)]["Sx"] for s in ss])) for e in EG}; es = max(mb, key=mb.get)
        tag = f"h{haz}_t{t + 1}"
        L.append(f"\n## haz = {haz}, {tl}{'  (PRIMARY)' if (haz == 0.004 and t == 299) else ''}\n\nn = {len(ss)} seeds {ss}. Fair search: across-seed mean S_BLIND(e): " + ", ".join(f"{e}: {mb[e]:.3f}" for e in EG) + f" → **e\\* = {es}**.\n\n" + H)
        rep[tag + "_estar"] = es; rep[tag + "_blind_means"] = mb
        r1 = row(tag + "_Dstar", f"**D\\* = S_COND − S_BLIND(e\\* = {es})**", [S[s][("blind", es)]["Sc"] - S[s][("blind", es)]["Sx"] for s in ss])
        r2 = row(tag + "_Dsham", "**D_sham = S_COND − S_SHAM**", [S[s][("sham", 0.0)]["Sc"] - S[s][("sham", 0.0)]["Sx"] for s in ss])
        row(tag + "_Dbrain", "S_COND − S_BRAIN (evolved brain eats; reported)", [S[s][("brain", 0.0)]["Sc"] - S[s][("brain", 0.0)]["Sx"] for s in ss])
        for e in EG:
            row(tag + f"_D{e}", f"S_COND − S_BLIND({e})", [S[s][("blind", e)]["Sc"] - S[s][("blind", e)]["Sx"] for s in ss])
        if t == 299:
            figdat[haz] = dict(EG=EG, blind=[[S[s][("blind", e)]["Sx"] for s in ss] for e in EG], cond=[[S[s][("blind", e)]["Sc"] for s in ss] for e in EG],
                               sham=[S[s][("sham", 0.0)]["Sx"] for s in ss], brain=[S[s][("brain", 0.0)]["Sx"] for s in ss])
        g = lambda k, c: np.mean([S[s][c][k] for s in ss])
        L.append("\n| group (in the run vs comparator) | survival | predation deaths | starvation deaths | mean eat effort | mean E of survivors |\n|---|---|---|---|---|---|")
        for c, nm in ((("blind", es), f"BLIND(e*={es})"), (("sham", 0.0), "SHAM"), (("brain", 0.0), "BRAIN")):
            L.append(f"| COND (vs {nm}) | {g('Sc', c):.3f} | {g('Pc', c):.3f} | {g('STc', c):.3f} | {g('eatc', c):.3f} | {g('Ec', c):.3f} |")
            L.append(f"| {nm} | {g('Sx', c):.3f} | {g('Px', c):.3f} | {g('STx', c):.3f} | {g('eatx', c):.3f} | {g('Ex', c):.3f} |")
        if haz == 0.004 and t == 299:
            ok = r1["mean"] >= MARGIN and r1["lo"] > 0 and r2["lo"] > 0
            rep["verdict"] = "PASS" if ok else "FAIL"
            rep["verdict_detail"] = dict(Dstar_mean_ge_margin=bool(r1["mean"] >= MARGIN), Dstar_lo_gt0=bool(r1["lo"] > 0), Dsham_lo_gt0=bool(r2["lo"] > 0))
            L.append(f"\n**Pre-registered verdict: {rep['verdict']}** ({rep['verdict_detail']}; margin +{MARGIN})")

# ---- pre-planned branch: assay positive control (hand-coded COND population), if run
AS = {}
for f in sorted(glob.glob(os.path.join(HERE, "runs25", "assay25_s*.pkl"))):
    d = pickle.load(open(f, "rb")); AS[d["seed"]] = d
if AS:
    ss = sorted(AS); lo = 75
    def rE(a, r, E0): return float((1.0 - (r["E"] - a["E"])[lo:].mean(0) / (0.5 * E0)).mean())
    PS = {s: {arm: rE(AS[s]["arms"][arm], AS[s]["arms"]["ref"], AS[s]["E0"]) for arm in ("cl", "open", "frozen")} for s in ss}
    valid = all(AS[s]["clamp0_maxdiff_E"] == 0 for s in ss); rep["assay_valid"] = valid
    L.append(f"\n## Assay positive control (hand-coded COND population, haz = 0.004, 50% drop, T = 150, births off; r_E, dead = 0)\n\nn = {len(ss)} seeds. Validity clamp0 == ref exactly: **{valid}** (max |dE| per seed: " + ", ".join(f"{AS[s]['clamp0_maxdiff_E']:.1e}" for s in ss) + ")\n\n" + H)
    rA = row("assay_A", "**A = r_CL − r_OPEN**", [PS[s]["cl"] - PS[s]["open"] for s in ss])
    rB = row("assay_B", "**B = r_CL − r_FROZEN**", [PS[s]["cl"] - PS[s]["frozen"] for s in ss])
    for arm in ("cl", "open", "frozen"):
        row(f"assay_r_{arm}", f"r_E {arm.upper()}", [PS[s][arm] for s in ss])
    for arm in ("ref", "cl", "open", "frozen"):
        row(f"assay_surv_{arm}", f"target survival at T, {arm}", [float(AS[s]["arms"][arm]["alive"][-1].mean()) for s in ss])
        row(f"assay_eat_{arm}", f"target mean eat effort, steps 0-30, {arm}", [float(AS[s]["arms"][arm]["eat"][:30].mean()) for s in ss])
    rep["assay_verdict"] = "DETECTS" if (valid and rA["lo"] > 0 and rB["lo"] > 0) else "DOES NOT DETECT"
    L.append(f"\n**Assay positive control: {rep['assay_verdict']}** the known feedback controller (both A and B CIs > 0 and validity exact required). Hand-coded rule: not an agency claim about any evolved population.")

# ---- post hoc power check (exploratory, specified in NIGHTLY_LOG 01:48 before running): K = 200 targets per seed
for SUF, KN in (("", 40), ("_k200", 200)):
    AS = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs25", f"assay25{SUF}_s*.pkl"))):
        d = pickle.load(open(f, "rb")); AS[d["seed"]] = d
    if not AS: continue
    ss = sorted(AS); lo = 75
    def rE(a, r, E0): return float((1.0 - (r["E"] - a["E"])[lo:].mean(0) / (0.5 * E0)).mean())
    PS = {s: {arm: rE(AS[s]["arms"][arm], AS[s]["arms"]["ref"], AS[s]["E0"]) for arm in ("cl", "open", "frozen")} for s in ss}
    SV = {s: {arm: float(AS[s]["arms"][arm]["alive"][-1].mean()) for arm in ("ref", "cl", "open", "frozen")} for s in ss}
    tag = "ph" + (SUF or "_k40")
    if SUF:
        valid = all(AS[s]["clamp0_maxdiff_E"] == 0 for s in ss); rep[tag + "_valid"] = valid
        L.append(f"\n## POST HOC power check: assay positive control with K = {KN} targets/seed (exploratory)\n\nn = {len(ss)} seeds, targets/seed: " + ", ".join(str(len(AS[s]['tg'])) for s in ss) + f". Validity clamp0 == ref: **{valid}**\n\n" + H)
        row(tag + "_A", "A = r_CL − r_OPEN", [PS[s]["cl"] - PS[s]["open"] for s in ss])
        row(tag + "_B", "B = r_CL − r_FROZEN", [PS[s]["cl"] - PS[s]["frozen"] for s in ss])
        for arm in ("cl", "open", "frozen"): row(f"{tag}_r_{arm}", f"r_E {arm.upper()}", [PS[s][arm] for s in ss])
        for arm in ("ref", "cl", "open", "frozen"): row(f"{tag}_eat_{arm}", f"target mean eat effort, steps 0-30, {arm}", [float(AS[s]["arms"][arm]["eat"][:30].mean()) for s in ss])
    else:
        L.append(f"\n## POST HOC: survival read-out of the K = 40 positive control (not pre-registered)\n\n" + H)
    row(tag + "_Asurv", "A_surv = surv_CL − surv_OPEN (target survival at T)", [SV[s]["cl"] - SV[s]["open"] for s in ss])
    row(tag + "_Bsurv", "B_surv = surv_CL − surv_FROZEN", [SV[s]["cl"] - SV[s]["frozen"] for s in ss])
    row(tag + "_dsurv", "surv_ref − surv_CL (cost of the drop)", [SV[s]["ref"] - SV[s]["cl"] for s in ss])
    pr = lambda arm: [float(AS[s]["arms"][arm]["pred"].any(0).mean()) for s in ss]
    for arm in ("ref", "cl", "open", "frozen"): row(f"{tag}_pred_{arm}", f"fraction of targets killed by predation, {arm}", pr(arm))
open(os.path.join(O, "tables25.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report25.json"), "w"), indent=1, default=float)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(figdat), figsize=(5.5 * len(figdat), 4.2), squeeze=False)
    for ax, haz in zip(axs[0], sorted(figdat, reverse=True)):
        f = figdat[haz]; EG = f["EG"]
        for arr, lab, col in ((f["blind"], "BLIND(e)", "tab:gray"), (f["cond"], "COND (same runs)", "tab:red")):
            m = [np.mean(a) for a in arr]; h = [ci(a)[2] - ci(a)[0] for a in arr]
            ax.errorbar(EG, m, yerr=h, marker="o", label=lab, color=col, capsize=2)
        for v, lab, col in ((f["sham"], "SHAM", "tab:blue"), (f["brain"], "BRAIN", "tab:green")):
            m, lo, hi, _ = ci(v); ax.axhspan(lo, hi, color=col, alpha=0.15); ax.axhline(m, color=col, lw=1, label=lab)
        ax.set_xlabel("blind constant eat effort e"); ax.set_ylabel("cohort survival at T = 300"); ax.set_title(f"haz = {haz}  (n = {len(f['sham'])}, 95% t CI)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(O, "exp25_paycheck.png"), dpi=120)
except Exception as ex:
    print("figure failed:", ex)
print("\n".join(L))
