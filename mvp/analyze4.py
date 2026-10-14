"""Analysis for Experiment 4 ("make regulation pay"). Reads runs4/price_*.pkl (or partial_*.pkl) and runs4/perturb/*.pkl.
Writes out4/: tables4.md, report4.json, price4_cumulative.png, lambda4_traj.png, perturb4_recovery.png, perturb4_arms.png."""
import os, glob, pickle, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs4"); O = os.path.join(HERE, "out4"); os.makedirs(O, exist_ok=True)
V = ["full", "shuffled", "no_cost", "static_env", "pe", "lamcost", "pe_lamcost"]
COL = dict(full="k", shuffled="tab:gray", no_cost="tab:olive", static_env="tab:brown", pe="tab:blue", lamcost="tab:red", pe_lamcost="tab:purple")
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return (float(x.mean()) if n else np.nan, np.nan, np.nan, n)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return (float(m), float(m - h), float(m + h), n)
f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
rep = dict(price={}, perturb={}, verdict={})
L = ["# Experiment 4: make regulation pay — tables (95% t-CI over seeds; n=3 → df=2)\n"]
price = {}
for v in V:
    p = os.path.join(R, f"price_{v}.pkl"); pp = os.path.join(R, f"partial_{v}.pkl")
    if os.path.exists(p): price[v] = (pickle.load(open(p, "rb")), "final")
    elif os.path.exists(pp): price[v] = (pickle.load(open(pp, "rb")), "partial")
for q, lab in (("a", "(a) λ under actual inputs (since birth)"), ("b", "(b) λ under standardized input bank")):
    L += [f"\n## Price decomposition {lab}\n", "| variant | status | t_end | Δλ̄ | S | T | W | I | |S+T|>|W| seeds | max cum resid |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, (d, stt) in price.items():
        ser = [{k: np.asarray(x) for k, x in s.items()} for s in d["series"]]
        cum = {k: [s[f"{q}_{k}"].sum() for s in ser] for k in "DSTWI"}
        res = max(abs(s[f"{q}_D"].sum() - sum(s[f"{q}_{k}"].sum() for k in "STWI")) for s in ser)
        dom = sum(abs(cum["S"][i] + cum["T"][i]) > abs(cum["W"][i]) for i in range(len(ser)))
        rep["price"].setdefault(v, {})[q] = dict(status=stt, t_end=int(ser[0]["t"][-1]), per_seed=cum, ci={k: ci(cum[k]) for k in cum}, sel_dom_seeds=int(dom), resid=float(res),
                                                 zbar_end=[float(s[f"{q}_zbar1"][-1]) for s in ser], pop_end=[int(s["pop"][-1]) for s in ser],
                                                 mean_cost_late=[float(s["mean_cost"][-50:].mean()) for s in ser])
        L.append(f"| {v} | {stt} | {ser[0]['t'][-1]} | " + " | ".join(f(ci(cum[k])) for k in "DSTWI") + f" | {dom}/{len(ser)} | {res:.1e} |")
L += ["\nλ̄ at end (a) per seed and population: " + "; ".join(f"{v}: λ̄ " + ", ".join(f"{z:+.3f}" for z in rep['price'][v]['a']['zbar_end']) + " pop " + ",".join(map(str, rep['price'][v]['a']['pop_end'])) for v in price)]
# figures: cumulative price + lambda trajectories
if price:
    fig, ax = plt.subplots(2, 5, figsize=(22, 8), sharex=True)
    for qi, q in enumerate("ab"):
        for ki, k in enumerate("DSTWI"):
            a = ax[qi, ki]
            for v, (d, _) in price.items():
                for si, s in enumerate(d["series"]):
                    a.plot(np.asarray(s["t"]), np.cumsum(s[f"{q}_{k}"]), color=COL[v], lw=1, alpha=.8, label=v if si == 0 else None)
            a.axhline(0, color="0.7", lw=.5); a.set_title(f"({q}) cumulative {k}")
    ax[0, 0].legend(fontsize=7); fig.suptitle("Exp4 Price decomposition of mean λ, 3 seeds per variant (lines = seeds)"); fig.tight_layout(); fig.savefig(os.path.join(O, "price4_cumulative.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for v, (d, _) in price.items():
        for si, s in enumerate(d["series"]):
            ax[0].plot(s["t"], s["a_zbar1"], color=COL[v], lw=1, label=v if si == 0 else None)
            ax[1].plot(s["t"], s["b_zbar1"], color=COL[v], lw=1)
            ax[2].plot(s["t"], s["pop"], color=COL[v], lw=1)
    for a, t in zip(ax, ["mean λ (actual inputs)", "mean λ (standard bank)", "population"]): a.set_title(t); a.set_xlabel("step")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(O, "lambda4_traj.png"), dpi=110); plt.close(fig)
# perturbation
P = {}
for p in sorted(glob.glob(os.path.join(R, "perturb", "pert_*.pkl"))):
    d = pickle.load(open(p, "rb")); P.setdefault(d["variant"], []).append(d)
CONDS = ["kick0.03", "kick0.1", "kick0.3", "lyap", "ongoing"]
L += ["\n## Perturbation recovery (per-seed means over agents, 95% t-CI over seeds)\n",
      "| variant | seeds | cond | n agents | log r_cl(50) | log r_ol(50) | log r_fr(50) | cl−ol @50 | cl−fr @50 | ongoing log(cl/ol) t50–100 | extra cost (100 steps) | ΔE(100) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
curves = {}
for v in V:
    if v not in P: continue
    for c in CONDS:
        ps = dict(cl=[], ol=[], fr=[], dco=[], dcf=[], og=[], xc=[], dE=[], n=0)
        cv = []
        for d in P[v]:
            r = d["conds"][c]; ok = r["alive_cl"][:, 50] & r["alive_ref"][:, 50]
            if c == "ongoing":
                d0 = np.full(len(ok), 0.01 * np.sqrt(16))
            else:
                d0 = r["d0"]
            lr = lambda x: np.log(np.maximum(x[:, 50], 1e-12) / d0)
            good = ok & np.isfinite(d0) & (d0 > 0)
            if good.sum() == 0: continue
            cl, ol, fr = lr(r["d_cl"])[good], lr(r["d_ol"])[good], lr(r["d_fr"])[good]
            ps["cl"].append(cl.mean()); ps["ol"].append(ol.mean()); ps["fr"].append(fr.mean()); ps["dco"].append((cl - ol).mean()); ps["dcf"].append((cl - fr).mean())
            okl = r["alive_cl"][:, 99] & r["alive_ref"][:, 99]
            if c == "ongoing" and okl.sum():
                ps["og"].append(np.log(r["d_cl"][okl, 50:100].mean(1) + 1e-12).mean() - np.log(r["d_ol"][okl, 50:100].mean(1) + 1e-12).mean())
            ps["xc"].append((r["cost_cl"].sum(1) - r["cost_ref"].sum(1))[okl].mean() if okl.sum() else np.nan)
            ps["dE"].append((r["E_cl"][:, 99] - r["E_ref"][:, 99])[okl].mean() if okl.sum() else np.nan)
            ps["n"] += int(good.sum())
            with np.errstate(divide="ignore"):
                cv.append(np.stack([np.nanmean(np.log(np.maximum(r[k][good], 1e-12) / d0[good, None]), 0) for k in ("d_cl", "d_ol", "d_fr")]))
        if not ps["cl"]: continue
        curves[(v, c)] = np.stack(cv)
        row = {k: ci(ps[k]) for k in ("cl", "ol", "fr", "dco", "dcf", "og", "xc", "dE")}; row["n"] = ps["n"]; row["seeds"] = len(ps["cl"])
        rep["perturb"].setdefault(v, {})[c] = dict(row, per_seed={k: [float(x) for x in ps[k]] for k in ("cl", "ol", "fr", "dco", "og", "xc")})
        L.append(f"| {v} | {row['seeds']} | {c} | {row['n']} | " + " | ".join(f(row[k]) if np.isfinite(row[k][0]) else "" for k in ("cl", "ol", "fr", "dco", "dcf", "og", "xc", "dE")) + " |")
    # verdict (Evelyn's rule)
    if v in rep["price"]:
        sel = rep["price"][v]["b"]["sel_dom_seeds"] == len(rep["price"][v]["b"]["per_seed"]["S"])
        kick = [rep["perturb"][v][c] for c in ("kick0.03", "kick0.1", "kick0.3") if c in rep["perturb"].get(v, {})]
        active = any(k["dco"][2] < 0 for k in kick) or ("ongoing" in rep["perturb"][v] and rep["perturb"][v]["ongoing"]["og"][2] < 0)
        passive = all(k["ol"][0] < 0 for k in kick)
        rec = "active > passive" if active else ("passive" if passive else "none")
        lab = {(True, "passive"): "stable inherited dynamics", (True, "none"): "population sorting", (False, "none"): "input-dependent dynamics",
               (False, "passive"): "input-dependent dynamics (passive contraction)"}.get((sel, rec), "individual regulation" if rec == "active > passive" else "?")
        rep["verdict"][v] = dict(selection_dominates=bool(sel), recovery=rec, label=lab)
L += ["\n## Decision rule\n", "| variant | selection dominates (|S+T|>|W| all seeds, trait b standardized inputs) | recovery | label |", "|---|---|---|---|"]
for v, d in rep["verdict"].items(): L.append(f"| {v} | {d['selection_dominates']} | {d['recovery']} | **{d['label']}** |")
# arms vs full, paired by seed
L += ["\n## Arms vs full (paired by seed; difference in cl−ol @50 and in ongoing log(cl/ol); negative = arm regulates better)\n", "| arm | cond | Δ(cl−ol) vs full | Δ ongoing vs full |", "|---|---|---|---|"]
for v in ("pe", "lamcost", "pe_lamcost", "shuffled", "no_cost", "static_env"):
    if v in rep["perturb"] and "full" in rep["perturb"]:
        for c in ("kick0.1", "ongoing"):
            a, b = rep["perturb"][v].get(c), rep["perturb"]["full"].get(c)
            if a and b and len(a["per_seed"]["dco"]) == len(b["per_seed"]["dco"]):
                dd = np.array(a["per_seed"]["dco"]) - np.array(b["per_seed"]["dco"])
                og = (np.array(a["per_seed"]["og"]) - np.array(b["per_seed"]["og"])) if c == "ongoing" else np.array([np.nan])
                L.append(f"| {v} | {c} | {f(ci(dd))} | {f(ci(og)) if c == 'ongoing' else ''} |")
open(os.path.join(O, "tables4.md"), "w").write("\n".join(L) + "\n")
json.dump(rep, open(os.path.join(O, "report4.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
if curves:
    vs = [v for v in V if (v, "kick0.1") in curves]
    fig, ax = plt.subplots(1, len(vs), figsize=(3.3 * len(vs), 3.6), sharey=True)
    ax = np.atleast_1d(ax)
    for a, v in zip(ax, vs):
        cv = curves[(v, "kick0.1")]
        for i, (nm, cc) in enumerate((("closed", "tab:red"), ("open-loop", "tab:blue"), ("frozen", "tab:green"))):
            for s in cv: a.plot(s[i], color=cc, lw=1, alpha=.8, label=nm if s is cv[0] else None)
        a.axhline(0, color="0.6", lw=.5); a.set_title(v); a.set_xlabel("step after kick (ε=0.1)")
    ax[0].set_ylabel("mean log d(t)/ε"); ax[0].legend(fontsize=7); fig.suptitle("Exp4 perturbation recovery (lines = seeds)"); fig.tight_layout(); fig.savefig(os.path.join(O, "perturb4_recovery.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (c, key, t) in enumerate((("kick0.1", "dco", "closed − open-loop log r @50 (kick 0.1)"), ("ongoing", "og", "ongoing: log(closed/open) t50–100"))):
        for i, v in enumerate(vs):
            r = rep["perturb"][v].get(c)
            if not r: continue
            m, lo, hi, _ = r[key]; ax[j].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[v], capsize=4)
            ax[j].scatter([i] * len(r["per_seed"][key]), r["per_seed"][key], color=COL[v], s=10, alpha=.5)
        ax[j].axhline(0, color="0.5", lw=.7); ax[j].set_xticks(range(len(vs))); ax[j].set_xticklabels(vs, rotation=30); ax[j].set_title(t)
    fig.suptitle("< 0 would mean the closed loop regulates better than passive replay (95% t-CI, n=3)"); fig.tight_layout(); fig.savefig(os.path.join(O, "perturb4_arms.png"), dpi=110); plt.close(fig)
print(open(os.path.join(O, "tables4.md")).read())
