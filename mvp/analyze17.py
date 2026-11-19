"""Exp 17 perturbation analysis (phase B, Exp 6 energy-drop protocol) on runs17p/p16B_*.pkl. The per-run metric function
phaseB_seed is taken verbatim from analyze14.py (exec of its source) so the scoring is identical to Exp 6/14.
PRIMARY (Exp 16 pre-registration, unchanged): B = r_cl - r_blind in sw_pays at t=3000, n=9 paired seeds, 95% t CI > 0.
Secondary: A = r_cl - r_clamp (active vs passive). Writes figures/exp17/perturb17.json and perturb17.md."""
import os, glob, pickle, json, re
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs17p"); F = os.path.join(HERE, "..", "figures", "exp17")
src = open(os.path.join(HERE, "analyze14.py")).read()
fn = src[src.index("def phaseB_seed"):src.index("EK = (")]
LATE = slice(60, 120); ecurves = {}; exec(fn)
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else float("nan"), float("nan"), float("nan"), n, float("nan"))
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return (float(m), float(m - h), float(m + h), n, float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else float("nan"))
out = {}; md = ["# Exp 17 perturbation (phase B: 50% energy drop, r = 1 − mean deficit over steps 60–120 / amount removed), t = 3000", "",
                "| arm | seeds done | r closed | r clamped | r blind | A = r_cl − r_clamp | B = r_cl − r_blind | per-seed B |", "|---|---|---|---|---|---|---|---|"]
for v in ("sw_pays", "sw_full", "sw_pays_blind"):
    per = {}
    for p in sorted(glob.glob(os.path.join(R, f"p16B_{v}_s*_t3000.pkl"))):
        d = pickle.load(open(p, "rb")); per[d["seed"]] = phaseB_seed(d); per[d["seed"]]["fork_check"] = d["fork_check"]
    if not per: continue
    seeds = sorted(per); g = lambda k: [float(per[s][k]) for s in seeds]
    row = dict(seeds=seeds, **{k: ci(g(k)) for k in ("r_cl", "r_clamp", "r_blind", "A", "Bv")}, per_seed_B=g("Bv"), per_seed_A=g("A"),
               fork_exact=all(per[s]["fork_check"]["exact"] for s in seeds), clamp0_max=max(float(per[s]["clamp0"]) for s in seeds))
    out[v] = row
    f = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]"
    md.append(f"| {v} | {len(seeds)} ({','.join(map(str, seeds))}) | {f(row['r_cl'])} | {f(row['r_clamp'])} | {f(row['r_blind'])} | {f(row['A'])} | {f(row['Bv'])} | " + " ".join(f"{x:+.4f}" for x in row["per_seed_B"]) + " |")
pr = out.get("sw_pays")
if pr:
    done = len(pr["seeds"]) == 9
    out["PRIMARY"] = dict(n=len(pr["seeds"]), complete=done, B=pr["Bv"], PASS=bool(done and pr["Bv"][1] > 0))
    md += ["", f"PRIMARY (sw_pays, n = {len(pr['seeds'])}/9): B = {pr['Bv'][0]:+.4f} [{pr['Bv'][1]:+.4f}, {pr['Bv'][2]:+.4f}] -> " +
           ("PASS" if out["PRIMARY"]["PASS"] else ("FAIL" if done else "INCOMPLETE (not scored)"))]
json.dump(out, open(os.path.join(F, "perturb17.json"), "w"), indent=1); open(os.path.join(F, "perturb17.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
