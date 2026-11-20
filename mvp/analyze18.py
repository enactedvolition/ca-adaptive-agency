"""Exp 18 analysis: reads runs18/*.json, writes figures/exp18/{report18.json,tables18.md,exp18_gate.png}."""
import os, json, glob, numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "runs18"); O = os.path.join(H, "..", "figures", "exp18"); os.makedirs(O, exist_ok=True)
def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean()
    if n < 2: return m, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return m, m - h, m + h
def load(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(R, pat))):
        d = json.load(open(f)); out[d["seed"]] = d
    return out
rep = {}; lines = ["# Exp 18 tables (aware THR 0.8 vs blind comparator; sim13 rep_fixed; 400 steps)\n"]
def block(name, d):
    if not d: return
    s = sorted(d); v = [d[k]["dshare"] for k in s]; m, lo, hi = ci(v)
    rep[name] = dict(seeds=s, dshare=v, share0=[d[k]["share0"] for k in s], shareT=[d[k]["shareT"] for k in s], mean=m, lo=lo, hi=hi, n=len(v), all_pos=bool(all(x > 0 for x in v)),
                     rate_aware=[d[k]["rate_A"] for k in s], rate_blind=[d[k]["rate_B"] for k in s], births_aware=[d[k]["births_A"] for k in s], births_blind=[d[k]["births_B"] for k in s])
    lines.append(f"\n## {name}  (n={len(v)})\nmean dshare {m:+.3f} [95% t CI {lo:+.3f}, {hi:+.3f}]; all positive: {rep[name]['all_pos']}\n\n| seed | share0 | shareT | dshare | aware rate | blind rate | births aware | births blind |\n|---|---|---|---|---|---|---|---|")
    for k in s:
        r = d[k]; lines.append(f"| {k} | {r['share0']:.3f} | {r['shareT']:.3f} | {r['dshare']:+.3f} | {r['rate_A']:.4f} | {r['rate_B']:.4f} | {r['births_A']:.0f} | {r['births_B']:.0f} |")
for ec in ("4.0", "1.0"):
    block(f"gate_ecap{ec}_p*", load(f"gate_ecap{ec}_swap0_s*.json")); block(f"swap_ecap{ec}_p*", load(f"gate_ecap{ec}_swap1_s*.json"))
    for p in (0.0025, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05): block(f"posthoc_ecap{ec}_p{p}", load(f"gate_ecap{ec}_swap0_p{p}_s*.json"))
sw = {}
for ec in ("4.0", "1.0"):
    for f in glob.glob(os.path.join(R, f"sweep_ecap{ec}_p*_s*.json")):
        d = json.load(open(f)); sw.setdefault(ec, {}).setdefault(d["p"], []).append(d["dshare"])
rep["sweep_means"] = {ec: {str(p): float(np.mean(v)) for p, v in sorted(m.items())} for ec, m in sw.items()}
open(os.path.join(O, "tables18.md"), "w").write("\n".join(lines) + "\n"); json.dump(rep, open(os.path.join(O, "report18.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
keys = [k for k in rep if k != "sweep_means"]; fig, ax = plt.subplots(figsize=(10, 0.35 * len(keys) + 1.5))
for i, k in enumerate(keys):
    r = rep[k]; ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]] if r["n"] > 1 else None, fmt="o", color="k"); ax.scatter(r["dshare"], [i] * r["n"], s=10, alpha=.5)
ax.axvline(0, color="r", lw=.8); ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7); ax.set_xlabel("aware dshare (mean, 95% t CI; dots = seeds)"); plt.tight_layout(); plt.savefig(os.path.join(O, "exp18_gate.png"), dpi=110)
for k in keys: r = rep[k]; print(k, r["n"], f"{r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]", r["all_pos"], np.round(r["dshare"], 3).tolist())
