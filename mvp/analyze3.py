"""Analysis for experiment 3: Price decomposition, selection gradients, perturbation recovery, decision rule."""
from __future__ import annotations
import os, sys, pickle, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "runs3"); OUT = os.path.join(HERE, "out3"); os.makedirs(OUT, exist_ok=True)
V = ["full", "shuffled", "no_cost", "no_mut", "static_env", "no_build"]
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
BG, FG = "#0b0d12", "#dde"

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return [float(np.mean(x)) if n else np.nan, np.nan, np.nan, n]
    m = x.mean(); h = TCRIT.get(n - 1, 1.96) * x.std(ddof=1) / np.sqrt(n)
    return [float(m), float(m - h), float(m + h), n]

def style(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=8, family="monospace", loc="left")
    ax.tick_params(colors="#99a", labelsize=7)
    for s in ax.spines.values(): s.set_color("#334")

def fmt(c, p=3):
    return f"{c[0]:+.{p}f} [{c[1]:+.{p}f}, {c[2]:+.{p}f}]"

# ------------------------------------------------------------------ Price
def price(D):
    rows, resid, perstep = {}, {}, {}
    for v in V:
        d = D[v]; rows[v] = {}
        for q in ("a", "b"):
            per = {k: [] for k in ("D", "S", "T", "W", "I", "Wy", "Wm", "res_cum", "res_max", "zstart", "zend")}
            for s in d["series"]:
                tot = {k: s[f"{q}_{k}"].sum() for k in ("D", "S", "T", "W", "I", "Wy", "Wm")}
                r = s[f"{q}_D"] - (s[f"{q}_S"] + s[f"{q}_T"] + s[f"{q}_W"] + s[f"{q}_I"])
                for k, x in tot.items(): per[k].append(x)
                per["res_cum"].append(abs(r.sum())); per["res_max"].append(np.abs(r).max())
                per["zstart"].append(s[f"{q}_zbar"][0]); per["zend"].append(s[f"{q}_zbar1"][-1])
                perstep.setdefault((v, q), []).append((s["t"], np.abs(r)))
            rows[v][q] = {k: ci(x) for k, x in per.items()}
            rows[v][q]["per_seed"] = {k: [float(y) for y in x] for k, x in per.items()}
            rows[v][q]["res_max_all"] = float(np.max(per["res_max"])); rows[v][q]["res_cum_all"] = float(np.max(per["res_cum"]))
            rows[v][q]["telescoping_err"] = float(np.max(np.abs(np.array(per["zend"]) - np.array(per["zstart"]) - np.array(per["D"]))))
        rows[v]["counts"] = dict(intervals=int(len(d["series"][0]["t"])), DT=d["DT"],
                                 births_per_seed=[int(s["n_off"].sum()) for s in d["series"]],
                                 immigrants_per_seed=[int(s["n_imm"].sum()) for s in d["series"]],
                                 orphans=[int(s["n_orphan"].sum()) for s in d["series"]], diag=d["diag"])
    # residual figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax, q, lab in zip(axes, ("a", "b"), ("(a) actual inputs", "(b) standardized input bank")):
        for v in V:
            for t, r in perstep[(v, q)]:
                ax.semilogy(t, np.maximum(r, 1e-20), lw=0.4, alpha=0.6)
        style(ax, f"|Δλ̄ − (S+T+W+I)| per {D['full']['DT']}-step interval, {lab}\nall 6 variants × 3 seeds; max {max(rows[v][q]['res_max_all'] for v in V):.1e}")
        ax.set_xlabel("step", color=FG)
    fig.tight_layout(); p = os.path.join(OUT, "price_residual.png"); fig.savefig(p, facecolor=BG, dpi=100); plt.close(fig)
    # cumulative components figure
    fig, axes = plt.subplots(2, 6, figsize=(20, 6.5), facecolor=BG, sharex=True)
    cols = dict(S="#ff6b6b", T="#ffd166", W="#4cc9f0", I="#aaa", D="#fff")
    for j, v in enumerate(V):
        for i, q in enumerate(("a", "b")):
            ax = axes[i, j]
            for s in D[v]["series"]:
                for k in ("S", "T", "W", "I", "D"):
                    ax.plot(s["t"], np.cumsum(s[f"{q}_{k}"]), color=cols[k], lw=1.4 if k == "D" else 0.9, alpha=0.8,
                            ls="--" if k == "D" else "-", label=k if s is D[v]["series"][0] else None)
            ax.axhline(0, color="#445", lw=0.5)
            style(ax, f"{v} ({q}) cumulative")
            if j == 0 and i == 0: ax.legend(fontsize=7, facecolor=BG, labelcolor=FG)
    fig.suptitle("Price decomposition of mean λ_max: cumulative S (selection), T (transmission/mutation), W (within-life), I (immigration); dashed = Δλ̄   [3 seeds overlaid]",
                 color=FG, family="monospace", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p2 = os.path.join(OUT, "price_cumulative.png"); fig.savefig(p2, facecolor=BG, dpi=90); plt.close(fig)
    return rows, [p, p2]

# ------------------------------------------------------------------ selection gradients
def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b[1:]

def gradients(D):
    out = {}
    M1 = ["lam", "rho", "wnorm", "gain", "lamb0"]
    M2 = M1 + ["eff", "act", "eat", "build"]
    for v in V:
        d = D[v]; out[v] = {}
        for which in ("snapshot", "lifetime"):
            per_seed = {m: [] for m in ("M1", "M2")}; uni = []; spear = []; ns = []; vif = []
            for b in range(len(d["jobs"])):
                t = {k: x[b] for k, x in d["tab"].items()}; ae = d["alive_end"][b]
                age = t["age"]
                tr = dict(lam=t["lam10"], rho=t["rho"], wnorm=t["wnorm"], gain=t["gain"], lamb0=t["lamb0"],
                          eff=t["c_int"] / np.maximum(t["c_cost"], 1e-9), act=t["c_act"] / np.maximum(age, 1),
                          eat=t["c_eat"] / np.maximum(age, 1), build=t["c_build"] / np.maximum(age, 1))
                if which == "snapshot":
                    ok = ae & (age >= 100); w = t["n_off"] / np.maximum(age, 1)
                else:
                    ok = (~ae) & (t["birth"] >= 200) & (t["birth"] <= 3000) & (age >= 30); w = t["n_off"]
                for k in M2: ok &= np.isfinite(tr[k])
                if no_cost_like(v) and np.nanstd(w[ok]) == 0: ok &= False
                ns.append(int(ok.sum()))
                if ok.sum() < 30 or np.std(w[ok]) == 0:
                    continue
                wr = w[ok] / w[ok].mean()
                Z = {k: (tr[k][ok] - tr[k][ok].mean()) / (tr[k][ok].std() + 1e-12) for k in M2}
                uni.append(float(np.mean(wr * Z["lam"])))
                ra = np.argsort(np.argsort(tr["lam"][ok])); rb = np.argsort(np.argsort(w[ok])); spear.append(float(np.corrcoef(ra, rb)[0, 1]))
                for m, keys in (("M1", M1), ("M2", M2)):
                    X = np.column_stack([Z[k] for k in keys]); per_seed[m].append(dict(zip(keys, ols(X, wr))))
                X = np.column_stack([Z[k] for k in M2]); C = np.corrcoef(X.T)
                try: vif.append(dict(zip(M2, np.diag(np.linalg.inv(C)).tolist())))
                except np.linalg.LinAlgError: pass
            res = dict(n_per_seed=ns, spearman_lam_fitness=dict(per_seed=spear, ci=ci(spear)),
                       S_lam=dict(per_seed=uni, ci=ci(uni)))
            for m, keys in (("M1", M1), ("M2", M2)):
                res[m] = {k: dict(per_seed=[p[k] for p in per_seed[m]], ci=ci([p[k] for p in per_seed[m]])) for k in keys}
            res["vif_max"] = {k: float(np.max([x[k] for x in vif])) for k in M2} if vif else {}
            out[v][which] = res
    return out

def no_cost_like(v):
    return False

# ------------------------------------------------------------------ perturbation
CONDS = ["kick0.03", "kick0.1", "kick0.3", "lyap", "ongoing"]
def perturb():
    P = {}
    for f in sorted(glob.glob(os.path.join(R3, "perturb", "pert_*.pkl"))):
        r = pickle.load(open(f, "rb")); P.setdefault(r["variant"], []).append(r)
    if not P: return None, [], {}
    stats = {}
    figs = []
    pv = [v for v in ["full", "shuffled", "no_cost", "static_env"] if v in P]
    fig, axes = plt.subplots(len(pv), 5, figsize=(22, 3.6 * len(pv)), facecolor=BG, squeeze=False)
    tt = np.arange(1, 101)
    for i, v in enumerate(pv):
        stats[v] = {}
        for j, c in enumerate(CONDS):
            ax = axes[i, j]
            seed_curves = {k: [] for k in ("cl", "ol", "fr")}
            agents = {k: [] for k in ("lr10", "lr50", "lr100", "fr50", "dlam", "dE", "surv_cl", "surv_ref", "dcost", "dact", "dpos", "ongo_cl", "ongo_ol", "ongo_fr", "ongo_lr_ol", "ongo_lr_fr",
                                      "r50_cl", "r50_ol", "r50_fr")}
            per_seed = {k: [] for k in agents}
            for r in P[v]:
                C = r["conds"][c]
                both = C["alive_cl"] & C["alive_ref"]
                if c == "ongoing":
                    base = np.full(C["d_cl"].shape[0], SIG_NORM)
                else:
                    base = C["d0"]
                lcl = np.log(np.maximum(C["d_cl"], 1e-12) / base[:, None])
                lol = np.log(np.maximum(C["d_ol"], 1e-12) / base[:, None])
                lfr = np.log(np.maximum(C["d_fr"], 1e-12) / base[:, None])
                lcl = np.where(both, lcl, np.nan)
                seed_curves["cl"].append(np.nanmean(lcl, 0)); seed_curves["ol"].append(np.nanmean(lol, 0)); seed_curves["fr"].append(np.nanmean(lfr, 0))
                a = dict(lr10=lcl[:, 9] - lol[:, 9], lr50=lcl[:, 49] - lol[:, 49], lr100=lcl[:, 99] - lol[:, 99], fr50=lcl[:, 49] - lfr[:, 49],
                         r50_cl=lcl[:, 49], r50_ol=lol[:, 49], r50_fr=lfr[:, 49],
                         dlam=np.nanmean(np.where(both, C["lam_cl"] - C["lam_ref"], np.nan)[:, 20:], 1),
                         dE=np.where(both[:, -1], C["E_cl"][:, -1] - C["E_ref"][:, -1], np.nan),
                         surv_cl=C["alive_cl"][:, -1].astype(float), surv_ref=C["alive_ref"][:, -1].astype(float),
                         dcost=np.nansum(np.where(both, C["cost_cl"] - C["cost_ref"], np.nan), 1),
                         dact=np.nanmean(np.where(both, C["d_act"], np.nan), 1), dpos=np.where(both[:, -1], C["d_pos"][:, -1], np.nan),
                         ongo_cl=np.nanmean(np.where(both, C["d_cl"], np.nan)[:, 50:], 1), ongo_ol=C["d_ol"][:, 50:].mean(1), ongo_fr=C["d_fr"][:, 50:].mean(1))
                a["ongo_lr_ol"] = np.log(np.maximum(a["ongo_cl"], 1e-12) / np.maximum(a["ongo_ol"], 1e-12))
                a["ongo_lr_fr"] = np.log(np.maximum(a["ongo_cl"], 1e-12) / np.maximum(a["ongo_fr"], 1e-12))
                for k, x in a.items():
                    agents[k].append(x); per_seed[k].append(float(np.nanmean(x)))
            st = {k: dict(seed_ci=ci(per_seed[k]), agent_mean=float(np.nanmean(np.concatenate(agents[k]))),
                          n_agents=int(np.isfinite(np.concatenate(agents[k])).sum())) for k in agents}
            st["n_seeds"] = len(P[v]); stats[v][c] = st
            for k, col, lab in (("cl", "#ff6b6b", "closed-loop"), ("ol", "#4cc9f0", "open-loop (recorded inputs)"), ("fr", "#ffd166", "frozen input")):
                Y = np.array(seed_curves[k]); m = Y.mean(0)
                if len(Y) > 1:
                    h = TCRIT.get(len(Y) - 1, 2) * Y.std(0, ddof=1) / np.sqrt(len(Y)); ax.fill_between(tt, m - h, m + h, color=col, alpha=0.2)
                ax.plot(tt, m, color=col, lw=1.4, label=lab)
            ax.axhline(0, color="#556", lw=0.6)
            base_lab = "log(d/σ_step)" if c == "ongoing" else "log(d(t)/ε)"
            style(ax, f"{v} · {c}: {base_lab}\nn={st['lr50']['n_agents']} agents/{len(P[v])} seeds; Δ(cl−ol)@50={st['lr50']['seed_ci'][0]:+.2f} [{st['lr50']['seed_ci'][1]:+.2f},{st['lr50']['seed_ci'][2]:+.2f}]")
            if i == 0 and j == 0: ax.legend(fontsize=7, facecolor=BG, labelcolor=FG)
            ax.set_xlabel("steps after perturbation", color=FG, fontsize=7)
    fig.suptitle("Paired perturbation-recovery curves: mean over agents of log hidden-state distance to the matched reference, 95% t-CI across seeds",
                 color=FG, family="monospace", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); p = os.path.join(OUT, "perturbation_recovery.png"); fig.savefig(p, facecolor=BG, dpi=85); plt.close(fig)
    figs.append(p)
    # viability / cost figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), facecolor=BG)
    for ax, k, lab in zip(axes, ("dE", "dcost", "dlam", "dpos"), ("ΔE at t=100 (pert−ref)", "extra energy spent over 100 steps", "Δ local λ (t 20–100)", "trajectory divergence at t=100")):
        xs, labels = [], []
        for v in pv:
            for c in CONDS:
                xs.append(stats[v][c][k]["seed_ci"]); labels.append(f"{v[:6]}·{c}")
        y = np.arange(len(xs))
        ax.errorbar([x[0] for x in xs], y, xerr=[[x[0] - x[1] for x in xs], [x[2] - x[0] for x in xs]], fmt="o", color="#4cc9f0", ms=3, ecolor="#557")
        ax.axvline(0, color="#aaa", lw=0.6); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6, color=FG)
        style(ax, lab + " (seed mean, 95% CI)")
    fig.tight_layout(); p2 = os.path.join(OUT, "perturbation_viability_cost.png"); fig.savefig(p2, facecolor=BG, dpi=90); plt.close(fig)
    figs.append(p2)
    return stats, figs, P

SIG_NORM = 0.01 * 4.0     # expected ||N(0, 0.01^2 I_16)|| ≈ 0.01*sqrt(16)

def sig(c):    # CI excludes 0 -> sign
    if not np.isfinite(c[1]): return 0
    return -1 if c[2] < 0 else (1 if c[1] > 0 else 0)

def decide(price_rows, pstats):
    """Selection vs within-life: paired per seed under (a): |S+T| vs |W(age>=50)| must hold in ALL seeds.
    Recovery: active = closed-loop beats BOTH open-loop and frozen-input (95% CI across seeds) for some kick at t=50,
    or under ongoing disturbance (log mean-distance ratio cl/ol and cl/frozen CI < 0). passive = open-loop or frozen
    contracts (log r(50) CI < 0) without closed-loop being better. none = no contraction anywhere."""
    out = {}
    for v in V:
        a = price_rows[v]["a"]["per_seed"]
        st = np.abs(np.array(a["S"]) + np.array(a["T"])); wm = np.abs(np.array(a["Wm"]))
        sel_dom = bool(np.all(st > wm)); within_shift = bool(np.all(wm > st))
        rec = "untested"; detail = {}
        if pstats and v in pstats:
            ps = pstats[v]
            beats = {c: bool(sig(ps[c]["lr50"]["seed_ci"]) < 0 and sig(ps[c]["fr50"]["seed_ci"]) < 0) for c in ("kick0.03", "kick0.1", "kick0.3", "lyap")}
            worse = {c: bool(sig(ps[c]["lr50"]["seed_ci"]) > 0) for c in ("kick0.03", "kick0.1", "kick0.3", "lyap")}
            passive = {c: bool(sig(ps[c]["r50_ol"]["seed_ci"]) < 0 or sig(ps[c]["r50_fr"]["seed_ci"]) < 0) for c in ("kick0.03", "kick0.1", "kick0.3", "lyap")}
            on = ps["ongoing"]
            beats_on = bool(sig(on["ongo_lr_ol"]["seed_ci"]) < 0 and sig(on["ongo_lr_fr"]["seed_ci"]) < 0)
            detail = dict(closed_beats_controls=beats, closed_worse_than_open=worse, passive_contraction=passive, ongoing_closed_beats=beats_on)
            if any(beats.values()) or beats_on:
                rec = "active (closed-loop beats open-loop and frozen-input)"
            elif any(passive.values()):
                rec = "passive (brain contracts intrinsically; closed loop not better)"
            else:
                rec = "none"
        if rec.startswith("active"):
            row = "Active recovery exceeds passive controls -> evidence of individual regulation"
        elif rec == "untested":
            row = ("Selection dominates" if sel_dom else "Within-life shifts" if within_shift else "Mixed") + "; recovery not tested (variant outside Exp 2 set)"
        elif sel_dom and rec.startswith("passive"):
            row = "Selection dominates; passive recovery -> stable inherited dynamics"
        elif sel_dom:
            row = "Selection dominates; no controlled recovery -> population-level sorting"
        elif within_shift:
            row = "Within-life shifts; no controlled recovery -> input-dependent dynamics"
        else:
            row = "Mixed: selection vs within-life ordering differs across seeds"
        out[v] = dict(ST_abs_per_seed=st.tolist(), Wm_abs_per_seed=wm.tolist(), selection_dominates=sel_dom, within_life_shift=within_shift,
                      recovery=rec, recovery_detail=detail, row=row)
    return out

def main():
    D = {v: pickle.load(open(os.path.join(R3, f"price_{v}.pkl"), "rb")) for v in V if os.path.exists(os.path.join(R3, f"price_{v}.pkl"))}
    assert len(D) == 6, D.keys()
    rows, pf = price(D)
    grads = gradients(D)
    pstats, figs, P = perturb()
    dec = decide(rows, pstats)
    rep = dict(price=rows, gradients=grads, perturb=pstats, decision=dec,
               replay=dict(wall_s={v: D[v]["wall_s"] for v in V}, mismatches={v: D[v]["diag"]["verify_mismatch"] for v in V}),
               figs=pf + figs)
    json.dump(rep, open(os.path.join(OUT, "report3.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # markdown tables
    L = ["# Experiment 3 tables (auto-generated by analyze3.py)\n",
         f"Price interval = {D['full']['DT']} steps; cumulative over steps {int(D['full']['series'][0]['t'][0]) - D['full']['DT']}–4000; mean [95% t-CI] across n=3 seeds.\n"]
    for q, lab in (("a", "(a) λ under each agent's actual input stream (running estimate from birth)"), ("b", "(b) λ under the standardized input bank (4×40-step real input sequences, from current h)")):
        L += [f"\n## Price decomposition {lab}\n", "| variant | Δλ̄ | S (selection) | T (transmission) | W (within-life) | W age<50 | W age≥50 | I (immigrants) | max per-step resid | cum resid |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v in V:
            r = rows[v][q]
            L.append(f"| {v} | {fmt(r['D'])} | {fmt(r['S'])} | {fmt(r['T'])} | {fmt(r['W'])} | {fmt(r['Wy'])} | {fmt(r['Wm'])} | {fmt(r['I'])} | {r['res_max_all']:.1e} | {r['res_cum_all']:.1e} |")
    L += ["\n## Per-seed cumulative terms (seed 0, 1, 2)\n"]
    for q in ("a", "b"):
        for v in V:
            ps = rows[v][q]["per_seed"]
            L.append(f"- ({q}) {v}: " + "; ".join(f"{k} " + ", ".join(f"{x:+.3f}" for x in ps[k]) for k in ("D", "S", "T", "W", "Wm", "I")))
    for which in ("snapshot", "lifetime"):
        L += [f"\n## Selection gradients ({which}) — Lande–Arnold β on relative fitness, standardized traits; mean [95% t-CI] over seeds\n",
              "| variant | n/seed | Spearman(λ,w) | S_λ (univariate) | β_λ (M1: genome traits) | β_λ (M2: +behaviour) | strongest M2 term |", "|---|---|---|---|---|---|---|"]
        for v in V:
            g = grads[v][which]
            if not g["M2"]["lam"]["per_seed"]:
                L.append(f"| {v} | {g['n_per_seed']} | n/a | n/a | n/a | n/a | n/a |"); continue
            top = max(g["M2"], key=lambda k: abs(g["M2"][k]["ci"][0]))
            L.append(f"| {v} | {g['n_per_seed']} | {fmt(g['spearman_lam_fitness']['ci'])} | {fmt(g['S_lam']['ci'])} | {fmt(g['M1']['lam']['ci'])} | {fmt(g['M2']['lam']['ci'])} | {top} {fmt(g['M2'][top]['ci'])} |")
        L.append("\nFull M2 gradients (full variant): " + "; ".join(f"{k} {fmt(x['ci'])}" for k, x in grads['full'][which]['M2'].items()))
    if pstats:
        L += ["\n## Perturbation recovery (paired; per-seed means, 95% t-CI over seeds)\n",
              "| variant | condition | n agents | log r_cl(50) | log r_ol(50) | log r_fr(50) | cl−ol @10 | cl−ol @50 | cl−frozen @50 | ΔE(100) | extra cost | Δ local λ | surv cl/ref |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in pstats:
            for c in CONDS:
                s = pstats[v][c]
                L.append(f"| {v} | {c} | {s['lr50']['n_agents']} | {fmt(s['r50_cl']['seed_ci'],2)} | {fmt(s['r50_ol']['seed_ci'],2)} | {fmt(s['r50_fr']['seed_ci'],2)} | {fmt(s['lr10']['seed_ci'],2)} | {fmt(s['lr50']['seed_ci'],2)} | {fmt(s['fr50']['seed_ci'],2)} | {fmt(s['dE']['seed_ci'],3)} | {fmt(s['dcost']['seed_ci'],4)} | {fmt(s['dlam']['seed_ci'],3)} | {s['surv_cl']['seed_ci'][0]:.2f}/{s['surv_ref']['seed_ci'][0]:.2f} |")
            s = pstats[v]["ongoing"]
            L.append(f"| {v} | ongoing σ={0.01}/unit/step: mean d (t 50–100) cl / ol / frozen; log(cl/ol), log(cl/fr) | {s['ongo_cl']['n_agents']} | {s['ongo_cl']['seed_ci'][0]:.3f} / {s['ongo_ol']['seed_ci'][0]:.3f} / {s['ongo_fr']['seed_ci'][0]:.3f} | {fmt(s['ongo_lr_ol']['seed_ci'],2)} | {fmt(s['ongo_lr_fr']['seed_ci'],2)} | | | | {fmt(s['dE']['seed_ci'],3)} | {fmt(s['dcost']['seed_ci'],4)} | {fmt(s['dlam']['seed_ci'],3)} | {s['surv_cl']['seed_ci'][0]:.2f}/{s['surv_ref']['seed_ci'][0]:.2f} |")
    L += ["\n## Decision rule\n", "| variant | \\|S+T\\| (a), per seed | \\|W age≥50\\| (a), per seed | recovery | row |", "|---|---|---|---|---|"]
    for v in V:
        d = dec[v]; L.append(f"| {v} | {', '.join(f'{x:.2f}' for x in d['ST_abs_per_seed'])} | {', '.join(f'{x:.2f}' for x in d['Wm_abs_per_seed'])} | {d['recovery']} | {d['row']} |")
    open(os.path.join(OUT, "tables3.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
