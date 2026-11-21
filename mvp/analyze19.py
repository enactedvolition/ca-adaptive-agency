"""Exp 19 analysis: pre-registered primary D = dshare_TRUE - dshare_SHAM (implanted share, last 200 steps minus t=0), paired 95% t CI.
Writes figures/exp19/{tables19.md, report19.json, exp19_share.png}."""
import os, glob, json, pickle
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs19"); O = os.path.join(HERE, "..", "figures", "exp19"); os.makedirs(O, exist_ok=True)
from exp19_run import corr
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean()
    if n < 2: return float(m), float("nan"), float("nan"), n
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return float(m), float(m - h), float(m + h), n
per = {}
for arm in ("true", "sham"):
    for f in sorted(glob.glob(os.path.join(R, f"run19_{arm}_s*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for b, sd in enumerate(d["seeds"]):
            s = d["series"][b]; t = np.asarray(s["t"]); sh = np.asarray(s["share"]); stt = np.asarray(s["stats"]); ts = t[1:]; T = d["steps"]
            pool = lambda lo, hi, j: corr(stt[(ts > lo) & (ts <= hi)][:, 5 * j:5 * j + 5].sum(0))
            per.setdefault(sd, {})[arm] = dict(share0=float(sh[0]), shareT=float(sh[t > T - 200].mean()), dshare=float(sh[t > T - 200].mean() - sh[0]),
                popT=float(np.asarray(s["pop"])[t > T - 200].mean()), corr_all_last500=pool(T - 500, T, 0), corr_m_last500=pool(T - 500, T, 1), corr_m_first100=pool(0, 100, 1),
                corr_all_first100=pool(0, 100, 0), rep_rate_last500=float(stt[ts > T - 500][:, 1].sum() / stt[ts > T - 500][:, 0].sum()),
                SS_m=float(np.sum(s["m_S"])), D_m=float(np.sum(s["m_D"])), SS_a=float(np.sum(s["a_S"])), D_a=float(np.sum(s["a_D"])),
                maxres=float(max(np.abs(s["m_resid"]).max(), np.abs(s["a_resid"]).max())), share_curve=sh.round(4).tolist(), t=t.tolist())
seeds = sorted(sd for sd in per if "true" in per[sd] and "sham" in per[sd])
rep = dict(seeds=seeds, n=len(seeds)); L = ["# Exp 19 tables (implanted aware rule, TRUE vs SHAM sensing; sim13 rep_fixed; Exp 17 sw_pays t=3000 forks)\n"]
def row(name, f):
    v = [f(sd) for sd in seeds]; m, lo, hi, n = ci(v); rep[name] = dict(per_seed=v, mean=m, lo=lo, hi=hi, n=n, n_pos=int(sum(x > 0 for x in v if not np.isnan(x))))
    L.append(f"| {name} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} | {rep[name]['n_pos']}/{n} | " + " ".join(f"{x:+.3f}" for x in v) + " |")
L.append(f"n = {len(seeds)} seeds {seeds}\n\n| metric | mean [95% t CI] | n | # > 0 | per seed |\n|---|---|---|---|---|")
row("PRIMARY D = dshare_true - dshare_sham", lambda s: per[s]["true"]["dshare"] - per[s]["sham"]["dshare"])
row("dshare_true", lambda s: per[s]["true"]["dshare"]); row("dshare_sham", lambda s: per[s]["sham"]["dshare"])
row("share0", lambda s: per[s]["true"]["share0"])
row("corr(rep,E) all last500: true - sham", lambda s: per[s]["true"]["corr_all_last500"] - per[s]["sham"]["corr_all_last500"])
row("corr(rep,E) all last500 true", lambda s: per[s]["true"]["corr_all_last500"]); row("corr(rep,E) all last500 sham", lambda s: per[s]["sham"]["corr_all_last500"])
row("corr(rep,E) m=1 first100 true", lambda s: per[s]["true"]["corr_m_first100"]); row("corr(rep,E) m=1 last500 true", lambda s: per[s]["true"]["corr_m_last500"])
row("corr m=1 true: last500 - first100", lambda s: per[s]["true"]["corr_m_last500"] - per[s]["true"]["corr_m_first100"])
row("SS_m (Price selection on implant) true - sham", lambda s: per[s]["true"]["SS_m"] - per[s]["sham"]["SS_m"])
row("SS_m true", lambda s: per[s]["true"]["SS_m"]); row("SS_m sham", lambda s: per[s]["sham"]["SS_m"])
row("SS_a (c_esr/l0n) true - sham", lambda s: per[s]["true"]["SS_a"] - per[s]["sham"]["SS_a"])
row("rep rate last500 true", lambda s: per[s]["true"]["rep_rate_last500"]); row("rep rate last500 sham", lambda s: per[s]["sham"]["rep_rate_last500"])
row("pop last200 true - sham", lambda s: per[s]["true"]["popT"] - per[s]["sham"]["popT"])
rep["max_abs_price_resid"] = float(max(per[s][a]["maxres"] for s in seeds for a in ("true", "sham"))) if seeds else None
P = rep["PRIMARY D = dshare_true - dshare_sham"]; T = rep["dshare_true"]
rep["PASS"] = bool(len(seeds) >= 3 and P["lo"] > 0 and T["lo"] > 0)
L.append(f"\nmax |Price residual| = {rep['max_abs_price_resid']:.2e}\n\nPRE-REGISTERED PASS (D CI > 0 AND dshare_true CI > 0): {rep['PASS']}")
rep["per"] = {str(s): {a: {k: v for k, v in per[s][a].items() if k not in ("share_curve", "t")} for a in per[s]} for s in per}
open(os.path.join(O, "tables19.md"), "w").write("\n".join(L) + "\n"); json.dump(rep, open(os.path.join(O, "report19.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in seeds:
    for a, c in (("true", "C0"), ("sham", "C3")): ax[0].plot(per[s][a]["t"], per[s][a]["share_curve"], color=c, alpha=.6, lw=1, label=a if s == seeds[0] else None)
ax[0].set_xlabel("step"); ax[0].set_ylabel("implanted (m=1) share"); ax[0].legend(); ax[0].set_title("Exp 19: implanted aware rule, TRUE vs SHAM")
ks = [k for k in rep if isinstance(rep[k], dict) and "mean" in rep[k] and not k.startswith(("pop", "SS_a"))]
for i, k in enumerate(ks):
    r = rep[k]; ax[1].errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", color="k"); ax[1].scatter(r["per_seed"], [i] * len(r["per_seed"]), s=8, alpha=.5)
ax[1].axvline(0, color="r", lw=.8); ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels(ks, fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(O, "exp19_share.png"), dpi=110)
print("\n".join(L))
