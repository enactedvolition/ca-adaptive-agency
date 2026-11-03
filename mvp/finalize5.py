"""Finalize Exp 5: validate outputs, copy code + figures into the public repo, append README section and NIGHTLY_LOG entry,
run the smoke test, commit and push. Refuses to commit if anything is incomplete."""
import os, json, shutil, subprocess, sys, datetime, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = "/workspace/ca-adaptive-agency"
LOG = os.path.join(REPO, "NIGHTLY_LOG.md")
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
def log(txt):
    with open(LOG, "a") as fh: fh.write(txt + "\n")
    shutil.copy(LOG, "/workspace/ca-agency-viz/NIGHTLY_LOG.md")
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M PST")
rep = json.load(open(os.path.join(HERE, "out5", "report5.json")))
probs = []
for v in V:
    p = rep["price"].get(v)
    if not p or p["a"]["status"] != "final" or p["b"]["status"] != "final": probs.append(f"price {v} not final")
    elif p["b"]["resid"] > 1e-10 or p["a"]["resid"] > 1e-4: probs.append(f"price {v} residual too large")
    if v not in rep["perturb"] or rep["perturb"][v].get("kick0.1", {}).get("seeds", 0) < 3: probs.append(f"perturb {v} incomplete")
if probs:
    log(f"\n- {now}: Exp 5 finished computing but did NOT pass validation, so nothing was committed: {'; '.join(probs)}.")
    sys.exit(1)
fmt = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]"
verd = rep["verdict"]; P = rep["perturb"]
labels = {v: verd[v]["label"] for v in V}
any_active = [v for v in V if verd[v]["recovery"] == "active > passive"]
rows = "\n".join(f"  | {v} | {fmt(P[v]['kick0.1']['dco'])} | {fmt(P[v]['ongoing']['og'])} | {labels[v]} |" for v in V)
pairs = rep.get("pairs", {})
prow = "\n".join(f"  | {k.split(':')[0].replace('-', ' vs ')} | {k.split(':')[1]} | {fmt(d['dco'])} |" for k, d in pairs.items())
sw = rep.get("switch", {})
pe_late = {v: rep["price"][v]["a"]["mean_pe_late"] for v in V}
import numpy as np
pem = lambda v: float(np.mean(pe_late[v]))
if any_active:
    head = f"**Active recovery beat passive replay in: {', '.join(any_active)}.** Check the CIs below before calling it agency; n = 3."
else:
    head = "**Charging for prediction error did not produce regulation either.** No variant's closed loop beats passive (open-loop or frozen-input) replay, so the decision rule gives no agency label."
sec = f"""
## Experiment 5: charge energy for prediction error, in a changing world

New `sim5.py` knobs (with both off, `sim5` reproduces `sim4` exactly):
- `pe_cost`: energy cost of 0.01 × the instantaneous mean squared error of the agent's own next-input predictor (which reads its hidden state), charged every step after a 10-step warm-up. In calibration it added about a quarter to a third to per-step spend.
- `sw_period`: the resource layout cycles through 4 patch layouts every 300 steps, so food moves and fast re-adaptation pays.

Seven variants × 3 seeds, N = 2048 slots, 3000 steps: sw_full, sw_pe, sw_pecost, sw_pecost_pe (main arm), pecost_pe_static (no switching), sw_pecost_pe_shuffled, sw_no_cost. Same Price and perturbation protocols as Exp 4 (forks at step 2500).

{head}

- **Kick ε = 0.1, closed − open-loop log r(50), and ongoing noise log(closed/open), steps 50–100** (positive = closed loop worse):

  | variant | closed − open [95% CI] | ongoing [95% CI] | decision-rule label |
  |---|---|---|---|
{rows}

- **Paired contrasts by seed** (negative = first arm regulates better):

  | contrast | cond | Δ(closed − open) |
  |---|---|---|
{prow}

- Late prediction-error MSE per step: sw_full {pem('sw_full'):.3f}, sw_pecost_pe {pem('sw_pecost_pe'):.3f}, pecost_pe_static {pem('pecost_pe_static'):.3f}, shuffled {pem('sw_pecost_pe_shuffled'):.3f}.
- Price identity residual: max {max(rep['price'][v]['b']['resid'] for v in V):.1e} (standardized inputs), {max(rep['price'][v]['a']['resid'] for v in V):.1e} (actual inputs).

Figures: [figures/exp5](figures/exp5). Full tables, including the Price decomposition and post-switch re-adaptation: `figures/exp5/tables5.md`. Generated automatically by `mvp/finalize5.py` from `report5.json`.
"""
rd = open(os.path.join(REPO, "README.md")).read()
if "## Experiment 5" not in rd:
    rowline = "| Exp 5 (`sim5.py`, `exp5_*.py`) | Make regulation pay by charging for prediction error in a switching world | " + ("active recovery in " + ", ".join(any_active) if any_active else "**No.** Still passive recovery; no variant's closed loop beats replay") + " (see below) |"
    rd = rd.replace("| Exp 4 (`sim4.py`, `exp4_*.py`) | Make regulation pay: (a) agents sense their own prediction error, (b) instability costs energy | see below |",
                    "| Exp 4 (`sim4.py`, `exp4_*.py`) | Make regulation pay: (a) agents sense their own prediction error, (b) instability costs energy | **No** (see below) |\n" + rowline)
    rd = rd.replace("\n## Reproduce", sec + "\n## Reproduce")
    rd = rd.replace("../.venv/bin/python exp4_price.py bank && ../.venv/bin/python exp4_price.py run --procs 7 && ../.venv/bin/python exp4_perturb.py --procs 8 && ../.venv/bin/python analyze4.py",
                    "../.venv/bin/python exp4_price.py bank && ../.venv/bin/python exp4_price.py run --procs 7 && ../.venv/bin/python exp4_perturb.py --procs 8 && ../.venv/bin/python analyze4.py\n../.venv/bin/python exp5_price.py bank && ../.venv/bin/python exp5_price.py run --procs 5 && ../.venv/bin/python exp5_perturb.py --procs 5 && ../.venv/bin/python analyze5.py")
    open(os.path.join(REPO, "README.md"), "w").write(rd)
for fn in ("sim5.py", "exp5_price.py", "exp5_perturb.py", "analyze5.py", "run5_chain.sh", "finalize5.py"):
    shutil.copy(os.path.join(HERE, fn), os.path.join(REPO, "mvp", fn))
os.makedirs(os.path.join(REPO, "figures", "exp5"), exist_ok=True)
for fn in glob.glob(os.path.join(HERE, "out5", "*")):
    shutil.copy(fn, os.path.join(REPO, "figures", "exp5"))
t = subprocess.run([os.path.join(HERE, "..", ".venv", "bin", "python"), "-m", "pytest", "-q", "tests"], cwd=REPO, capture_output=True, text=True)
if t.returncode != 0:
    log(f"\n- {now}: Exp 5 results are in, but the smoke test failed, so nothing was committed. Results are in /workspace/ca-agency-viz/mvp/out5.")
    sys.exit(1)
summ = (f"Exp 5 (prediction-error cost in a switching world) finished. " + (f"Active recovery beat passive in {', '.join(any_active)}; check CIs." if any_active else "No variant shows active recovery beating passive replay; every one gets the same passive label as Exp 4.")
        + f" Main arm sw_pecost_pe closed−open kick0.1 = {fmt(P['sw_pecost_pe']['kick0.1']['dco'])}; label: {labels['sw_pecost_pe']}.")
log(f"\n- {now}: {summ} Committed to the public repo (README section, figures/exp5).")
g = lambda *a: subprocess.run(["git", "-c", "user.name=EV Campbell", "-c", "user.email=63378631+enactedvolition@users.noreply.github.com", *a], cwd=REPO, capture_output=True, text=True)
g("add", "README.md", "NIGHTLY_LOG.md", "mvp", "figures/exp5")
c = g("commit", "-m", "Exp 5: prediction-error energy cost in a switching environment; results, figures, nightly log")
p = g("push", "origin", "main")
if p.returncode != 0:
    log(f"  - push failed: {p.stderr.strip()[:300]}")
    sys.exit(1)
print("committed and pushed", c.stdout[:200])
