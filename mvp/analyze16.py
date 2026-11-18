"""Analysis for Experiment 16 (design pass: brain-gated reproduction, sim12 rep_free). Reads runs16/bitcheck.json, paycheck_fork.json,
paycheck_init.json, compete.json. Writes ../figures/exp16/: tables16.md, report16.json, exp16_paycheck.png, exp16_compete.png.
No evolution was run (the pre-registered pay-check failed)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs16"); O = os.path.join(HERE, "..", "figures", "exp16"); os.makedirs(O, exist_ok=True)
L = lambda f: json.load(open(os.path.join(R, f)))
bc, pf, pi, cp = L("bitcheck.json"), L("paycheck_fork.json"), L("paycheck_init.json"), L("compete.json")
THRS = ["0.8", "1.2", "1.6"]
pc = lambda x: f"{100*x:+.1f}%"
lines = ["# Experiment 16 tables", "", f"Bit-for-bit (rep_free=0 vs sim11 all fields, sim9 all fields, seeds 0-4 x 300 steps): **{bc['bit_for_bit']}**", ""]
rep = dict(bit_for_bit=bc["bit_for_bit"], paycheck={}, compete={})
for nm, d in (("fork (pre-registered primary)", pf), ("init (secondary)", pi)):
    lines += [f"## Pay-check, start = {nm}", "", "| thr | pop aware | pop blind | Δpop (per seed) | Δnet intake (per seed) | births/step aware / blind | mean E aware / blind |", "|---|---|---|---|---|---|---|"]
    for t in THRS:
        s = d["summary"][t]; per = d["per"]
        ba = np.mean([per[k][f"aware{t}"]["births"] for k in per]); bb = np.mean([per[k][f"blind{t}"]["births"] for k in per])
        ea = np.mean([per[k][f"aware{t}"]["E"] for k in per]); eb = np.mean([per[k][f"blind{t}"]["E"] for k in per])
        lines.append(f"| {t} | {s['pop_aware']:.0f} | {s['pop_blind']:.0f} | {pc(s['pop_rel_mean'])} ({', '.join(pc(x) for x in s['pop_rel'])}) | {pc(s['net_rel_mean'])} ({', '.join(pc(x) for x in s['net_rel'])}) | {ba:.2f} / {bb:.2f} | {ea:.2f} / {eb:.2f} |")
    br = [d["per"][k]["brain"] for k in d["per"]]; of = [d["per"][k]["off"] for k in d["per"]]
    lines += ["", f"Unmodified brains, rep_free=1: pop {', '.join(f'{x['pop']:.0f}' for x in br)}, mean E {', '.join(f'{x['E']:.2f}' for x in br)}, rep>0 {', '.join(f'{x['rep']:.2f}' for x in br)}; rep_free=0: pop {', '.join(f'{x['pop']:.0f}' for x in of)}.",
              f"Best thr by population: {d['best_thr']}. **PASS (pre-registered): {d['PASS_prereg']}**; PASS (literal suggested): {d['PASS_literal_suggested']}", ""]
    rep["paycheck"][d["start"]] = dict(summary=d["summary"], best=d["best_thr"], PASS_prereg=d["PASS_prereg"], PASS_literal=d["PASS_literal_suggested"])
lines += ["## Mixed competition (post hoc / exploratory): aware lineages (even root) vs always-reproduce (odd root)", "", "| thr | share t=0 | share last 50 | Δshare | births aware / always |", "|---|---|---|---|---|"]
for t in THRS:
    for r in cp["per"][t]:
        lines.append(f"| {t} (s{r['seed']}) | {r['share0']:.3f} | {r['shareT']:.3f} | {r['dshare']:+.3f} | {r['births_aware']:.0f} / {r['births_always']:.0f} |")
    rep["compete"][t] = dict(dshare=[r["dshare"] for r in cp["per"][t]], mean=float(np.mean([r["dshare"] for r in cp["per"][t]])))
open(os.path.join(O, "tables16.md"), "w").write("\n".join(lines) + "\n"); json.dump(rep, open(os.path.join(O, "report16.json"), "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for j, (d, lab) in enumerate(((pf, "start: sw_full t=3000 forks"), (pi, "start: initial population"))):
    x = np.arange(3); w = 0.38
    for k, (key, c) in enumerate((("pop_rel", "C0"), ("net_rel", "C3"))):
        vals = np.array([d["summary"][t][key] for t in THRS])
        ax[j].bar(x + (k - .5) * w, 100 * vals.mean(1), w, color=c, alpha=.6, label="Δ population" if k == 0 else "Δ net intake / agent")
        for i in range(3): ax[j].scatter(np.full(3, x[i] + (k - .5) * w), 100 * vals[i], color=c, s=10, zorder=3)
    ax[j].axhline(5, ls=":", c="k", lw=.8); ax[j].axhline(0, c="k", lw=.6); ax[j].set_xticks(x); ax[j].set_xticklabels([f"thr {t}" for t in THRS])
    ax[j].set_ylabel("aware vs matched blind (%)"); ax[j].set_title(lab, fontsize=9)
ax[0].legend(fontsize=8); fig.suptitle("Exp 16 pay-check: brain-gated reproduction (rep_free=1)", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(O, "exp16_paycheck.png"), dpi=110)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for i, t in enumerate(THRS):
    for r in cp["per"][t]:
        ax.plot(np.arange(len(r["share_curve"])) * 10, r["share_curve"], c=f"C{i}", lw=1, label=f"aware thr {t}" if r["seed"] == 0 else None)
ax.set_xlabel("step"); ax.set_ylabel("share of aware lineages"); ax.set_title("Mixed competition vs always-reproduce (3 seeds each)", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(O, "exp16_compete.png"), dpi=110)
print("\n".join(lines))
