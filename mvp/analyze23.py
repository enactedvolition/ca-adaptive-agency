"""Exp 23 analysis: per-channel decomposition of A = rE_CL - rE_OPEN (births off, own E). PRIMARY (Exp 20 TRUE t=1500): A_move, A_eat, A_soc (r_CL - r_X),
full A, interaction I = A - (A_move+A_eat+A_soc); secondary A_moveeat, ctrl and sham pops. Validity: CL/OPEN reproduce Exp 22 bit-for-bit, clamp0 of every partial
arm == ref, 0 target births. Writes figures/exp23/{tables23.md, report23.json, exp23_channels.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "..", "figures", "exp23"); os.makedirs(O, exist_ok=True)
import exp20_run as X
ARM = ("twin", "cl", "full", "move", "eat", "soc", "moveeat")
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean() if n else float("nan")
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
def per_seed(d):
    A = d["arms"]; ref = A["ref"]; E0 = d["E0"]; o = {}
    for arm in ARM:
        a = d["twin"] if arm == "twin" else A[arm]; o[arm] = float(X.recovery(a["E"], a["passed"], ref["E"], ref["passed"], E0, offspring=False).mean())
    o["Eend"] = {arm: float(A[arm]["E"][-1].mean()) for arm in ("ref", "cl", "full", "move", "eat", "soc")}
    o["surv"] = {arm: float(A[arm]["alive"][-1].mean()) for arm in ("ref", "cl", "full", "move", "eat", "soc")}
    o["valid"] = bool(d["ref_matches_exp22"] and all(d["match22"].values()) and all(v == 0 for v in d["clamp0_maxdiff_E"].values()) and sum(d["births_total"].values()) == 0)
    o["detail"] = dict(ref_matches=d["ref_matches_exp22"], match22=d["match22"], clamp0=d["clamp0_maxdiff_E"], births=sum(d["births_total"].values())); return o
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "runs23", "run23_*_s*.pkl"))):
    d = pickle.load(open(f, "rb")); R.setdefault(d["pop"], {})[d["seed"]] = per_seed(d)
rep = dict(per_seed=R); L = ["# Exp 23 tables (births off; own-E r_E; partial-OPEN replay arms: MOVE a0-1, EAT a2, SOC a4-7, MOVEEAT a0-2; rest closed loop)\n"]
H = "| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|"
def row(name, f, ss):
    v = [f(s) for s in ss]; m, lo, hi, n = ci(v); rep[name] = dict(seeds=ss, per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v)))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |"); return rep[name]
verd = {}
for pop, ttl in (("true", "PRIMARY: Exp 20 TRUE (implant) t = 1500 forks"), ("ctrl", "SECONDARY: no-implant control (Exp 17 sw_pays t = 3000)"), ("sham", "SECONDARY: age-matched SHAM t = 1500")):
    P = R.get(pop, {}); ss = sorted(P)
    if not ss: continue
    valid = all(P[s]["valid"] for s in ss); rep[f"{pop}_valid_all"] = valid
    L.append(f"\n## {ttl}\n\nn = {len(ss)} seeds {ss}. Validity (CL & OPEN == Exp 22 bit-for-bit; clamp0 of all partial arms == ref; 0 target births): **{valid}**"
             + ("" if valid else f"  INVALID seeds: {[s for s in ss if not P[s]['valid']]}") + "\n\n" + H)
    r = lambda arm: (lambda s: P[s][arm])
    Af = row(f"{pop}: full A = CL - OPEN", lambda s: P[s]["cl"] - P[s]["full"], ss)
    Am = row(f"{pop}: A_move = CL - MOVE", lambda s: P[s]["cl"] - P[s]["move"], ss); Ae = row(f"{pop}: A_eat = CL - EAT", lambda s: P[s]["cl"] - P[s]["eat"], ss)
    As = row(f"{pop}: A_soc = CL - SOC", lambda s: P[s]["cl"] - P[s]["soc"], ss)
    row(f"{pop}: A_moveeat = CL - MOVEEAT", lambda s: P[s]["cl"] - P[s]["moveeat"], ss)
    row(f"{pop}: sum A_move + A_eat + A_soc", lambda s: 3 * P[s]["cl"] - P[s]["move"] - P[s]["eat"] - P[s]["soc"], ss)
    Ii = row(f"{pop}: interaction I = A - sum", lambda s: (P[s]["cl"] - P[s]["full"]) - (3 * P[s]["cl"] - P[s]["move"] - P[s]["eat"] - P[s]["soc"]), ss)
    row(f"{pop}: A_moveeat - A_move - A_eat (move x eat)", lambda s: (P[s]["cl"] - P[s]["moveeat"]) - (P[s]["cl"] - P[s]["move"]) - (P[s]["cl"] - P[s]["eat"]), ss)
    for arm in ARM: row(f"{pop}: rE_{arm}", r(arm), ss)
    row(f"{pop}: twin noise |1 - rE_twin|", lambda s: abs(1 - P[s]["twin"]), ss)
    for arm in ("ref", "cl", "full", "move", "eat", "soc"): row(f"{pop}: mean target E at T, {arm}", lambda s, a=arm: P[s]["Eend"][a], ss)
    for arm in ("ref", "cl", "full", "move", "eat", "soc"): row(f"{pop}: target survival at T, {arm}", lambda s, a=arm: P[s]["surv"][a], ss)
    fp = lambda c: bool(valid and c["lo"] > 0)
    if not valid: v = "VOID (validity failed)"
    elif fp(Am) or fp(Ae): v = "FORAGING-RELEVANT closed-loop correction present: " + ", ".join(n for n, c in (("move", Am), ("eat", Ae)) if fp(c))
    elif fp(As) and not (fp(Am) or fp(Ae)): v = "A carried by SOCIAL/BUILD replay (A_move, A_eat CIs include 0) - assay OPEN arm artifact"
    elif Af["lo"] > 0: v = "A not attributable to a single channel (interaction)"
    else: v = "no full A"
    verd[pop] = v; rep[f"{pop}_verdict"] = v; L.append(f"\n**{pop} verdict (pre-registered rule): {v}**\n")
rep["verdict"] = verd.get("true", "no true"); L.append(f"\n**Pre-registered primary verdict (TRUE): {rep['verdict']}**\n")
open(os.path.join(O, "tables23.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report23.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True); nm = ["full A", "A_move", "A_eat", "A_soc", "A_moveeat", "I"]
for j, pop in enumerate(("true", "ctrl", "sham")):
    P = R.get(pop, {}); ss = sorted(P)
    if not ss: continue
    cols = [lambda s: P[s]["cl"] - P[s]["full"], lambda s: P[s]["cl"] - P[s]["move"], lambda s: P[s]["cl"] - P[s]["eat"], lambda s: P[s]["cl"] - P[s]["soc"], lambda s: P[s]["cl"] - P[s]["moveeat"],
            lambda s: (P[s]["cl"] - P[s]["full"]) - (3 * P[s]["cl"] - P[s]["move"] - P[s]["eat"] - P[s]["soc"])]
    for i, c in enumerate(cols):
        v = [c(s) for s in ss]; ax[j].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=14); m, lo, hi, _ = ci(v); ax[j].errorbar(i + .3, m, [[m - lo], [hi - m]], fmt="ks", ms=4)
    ax[j].axhline(0, c="gray", lw=.5); ax[j].set_xticks(range(6)); ax[j].set_xticklabels(nm, fontsize=8); ax[j].set_title(f"Exp 23 {pop}: r_CL - r_X (births off)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(O, "exp23_channels.png"), dpi=110); print("\n".join(L))
