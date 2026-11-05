"""Finalize Exp 7: validate outputs, copy code + figures into the public repo, append README section and NIGHTLY_LOG entry,
run the smoke test, commit and push. Refuses to commit if anything is incomplete."""
import os, json, shutil, subprocess, sys, datetime, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = "/workspace/ca-adaptive-agency"
LOG = os.path.join(REPO, "NIGHTLY_LOG.md")
V = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
def log(txt):
    with open(LOG, "a") as fh: fh.write(txt + "\n")
    shutil.copy(LOG, "/workspace/ca-agency-viz/NIGHTLY_LOG.md")
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M PST")
rep_path = os.path.join(HERE, "out7", "report7.json")
if not os.path.exists(rep_path):
    log(f"\n- {now}: Exp 7 finished computing but report7.json is missing, so nothing was committed.")
    sys.exit(1)
rep = json.load(open(rep_path))
probs = []
n_pkl = len(glob.glob(os.path.join(HERE, "runs7", "dark_*.pkl")))
if n_pkl < 21: probs.append(f"only {n_pkl}/21 dark_*.pkl checkpoints")
for v in V:
    if v not in rep.get("regimes", {}):
        probs.append(f"{v} missing from regimes"); continue
    n = rep["regimes"][v]["dark_any"][3]
    if n < 3: probs.append(f"{v} has fewer than 3 seeds ({n})")
    for key in ("dark_any", "self_built", "other_built", "natural", "explore", "settled_noisy"):
        m = rep["regimes"][v][key][0]
        if not (m == m) or m < -1e-9 or m > 1.0 + 1e-9:
            probs.append(f"{v}.{key} out of [0,1]: {m}")
    # fractions should roughly sum to 1
    ssum = sum(rep["regimes"][v][k][0] for k in ("explore", "settled_noisy", "natural", "other_built", "self_built"))
    if abs(ssum - 1.0) > 0.05:
        probs.append(f"{v} regime means sum to {ssum:.3f}, not ~1")
if probs:
    log(f"\n- {now}: Exp 7 finished computing but did NOT pass validation, so nothing was committed: {'; '.join(probs)}.")
    sys.exit(1)
fmt = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]" if (len(c) > 2 and c[1] == c[1]) else f"{c[0]:+.4f}"
R = rep["regimes"]; S = rep["summaries"]; C = rep["contrasts"]; verd = rep["verdict"]
m = "sw_pecost_pe"
self_arms = verd["self_built_under_pecost_with_depletion"]
appear = [v for v, d in self_arms.items() if d["appears"]]
if appear:
    head = (f"**Self-built dark rooms appear under pe_cost with depletion still on in: {', '.join(appear)}.** "
            "Means and CIs are small (n = 3); see table. This is a dark-rooming / niche-construction result under the Friston RQ3 re-run, **not** an agency claim.")
else:
    head = ("**Self-built dark rooms do not appear under pe_cost while depletion is still on** "
            "(mean self-built fraction is 0 across pe_cost+depletion arms, or no seed shows any). "
            "Dark-rooming remains gated the way v2 found it: it shows up when starvation pressure is removed (see sw_no_cost), not when prediction error is charged. "
            "This is a dark-rooming result, **not** an agency claim.")
rows = "\n".join(
    f"  | {v} | {fmt(R[v]['explore'])} | {fmt(R[v]['settled_noisy'])} | {fmt(R[v]['natural'])} | {fmt(R[v]['other_built'])} | {fmt(R[v]['self_built'])} | {fmt(R[v]['dark_any'])} |"
    for v in V)
crows = "\n".join(
    f"  | {a} − {b} | {fmt(C[f'{a}_vs_{b}']['dark_any']['diff'])} | {fmt(C[f'{a}_vs_{b}']['self_built']['diff'])} | {fmt(C[f'{a}_vs_{b}']['other_built']['diff'])} | {fmt(C[f'{a}_vs_{b}']['natural']['diff'])} | {fmt(C[f'{a}_vs_{b}']['mean_surp']['diff'])} | {fmt(C[f'{a}_vs_{b}']['wall_mass']['diff'])} |"
    for a, b in (("sw_pecost_pe", "sw_full"), ("sw_pecost_pe", "sw_pecost_pe_shuffled"), ("sw_pecost_pe", "pecost_pe_static"), ("sw_pecost_pe", "sw_no_cost"))
    if f"{a}_vs_{b}" in C)
sec = f"""
## Experiment 7: dark rooms under pe_cost (RQ3 re-run on Exp 5 forks)

ARCHITECTURE.md §15 after "make PE matter": rerun RQ3 — do self-built dark rooms appear under `pe_cost` while depletion is still on (the Friston question). Exp 5 built `pe_cost` but only ran Price + hidden-state recovery; Exp 6 changed the recovery yardstick and still got nulls. More seeds on Exp 6 would only tighten intervals around a consistent null, so tonight classifies every alive agent on the existing Exp 5 forks (step 2500).

- Same 7 variants × 3 seeds. Regime logic is identical to `metrics2.regimes` (v2 thresholds): resident = residence ≥ 100 and speed < 0.2 v_term; quiet via analytic OU variance; dark natural / other-built / self-built by wall counterfactuals. `env_var = 0.35`, `build_eff = 1.0`, layout index at t = 2500 with period 300 is 0.
- Paired contrasts by seed for the main arm `sw_pecost_pe` vs `sw_full`, `sw_pecost_pe_shuffled`, `pecost_pe_static`, `sw_no_cost`.
- Dark-rooming is a separate research question. **No agency claim is made here.**

{head}

  | variant | explore | settled-noisy | dark: natural | dark: other-built | dark: self-built | dark_any |
  |---|---|---|---|---|---|---|
{rows}

  | contrast | Δ dark_any | Δ self-built | Δ other-built | Δ natural | Δ mean surp | Δ wall mass |
  |---|---|---|---|---|---|---|
{crows}

Main arm `{m}`: self-built {fmt(R[m]['self_built'])}, dark_any {fmt(R[m]['dark_any'])}, mean surp {fmt(S[m]['mean_surp'])}, wall mass {fmt(S[m]['wall_mass'])}. Full tables: `figures/exp7/tables7.md`. Figures: [figures/exp7](figures/exp7). Generated by `mvp/finalize7.py` from `report7.json`.
"""
rd = open(os.path.join(REPO, "README.md")).read()
if "## Experiment 7" not in rd:
    ans = ("**Partly:** self-built dark rooms under pe_cost+depletion in " + ", ".join(appear)) if appear else "**No.** Self-built dark rooms do not appear under pe_cost while depletion is on (see below)"
    rowline = f"| Exp 7 (`exp7_darkroom.py`) | RQ3 re-run: do self-built dark rooms appear under pe_cost with depletion still on? | {ans} |"
    anchor = [l for l in rd.split("\n") if l.startswith("| Exp 6 (`sim6.py`")][0]
    rd = rd.replace(anchor, anchor + "\n" + rowline)
    rd = rd.replace("\n## Reproduce", sec + "\n## Reproduce")
    l6 = "../.venv/bin/python exp6_regulate.py --procs 6 && ../.venv/bin/python analyze6.py   # needs the Exp 5 forks in runs5/"
    rd = rd.replace(l6, l6 + "\n../.venv/bin/python exp7_darkroom.py --procs 6 && ../.venv/bin/python analyze7.py   # needs the Exp 5 forks in runs5/")
    open(os.path.join(REPO, "README.md"), "w").write(rd)
for fn in ("exp7_darkroom.py", "analyze7.py", "run7_chain.sh", "finalize7.py", "finalize7.sh"):
    shutil.copy(os.path.join(HERE, fn), os.path.join(REPO, "mvp", fn))
os.makedirs(os.path.join(REPO, "figures", "exp7"), exist_ok=True)
for fn in glob.glob(os.path.join(HERE, "out7", "*")):
    shutil.copy(fn, os.path.join(REPO, "figures", "exp7"))
t = subprocess.run([os.path.join(HERE, "..", ".venv", "bin", "python"), "-m", "pytest", "-q", "tests"], cwd=REPO, capture_output=True, text=True)
if t.returncode != 0:
    log(f"\n- {now}: Exp 7 results are in, but the smoke test failed, so nothing was committed. Results are in /workspace/ca-agency-viz/mvp/out7.\n  pytest: {(t.stdout + t.stderr)[:400]}")
    sys.exit(1)
sb = R[m]["self_built"]; da = R[m]["dark_any"]
summ = (f"Exp 7 (dark-room classification on Exp 5 forks) finished. {verd['statement']} "
        f"Main arm {m}: self-built {fmt(sb)}, dark_any {fmt(da)}, mean surp {fmt(S[m]['mean_surp'])}, wall mass {fmt(S[m]['wall_mass'])}. "
        f"sw_no_cost dark_any {fmt(R['sw_no_cost']['dark_any'])}, self-built {fmt(R['sw_no_cost']['self_built'])}.")
next_note = (" Next step, and why: if self-built stayed at zero under pe_cost+depletion, the Friston-style PE cost did not open niche construction while starvation is on; "
             "a useful follow-up is a true counterfactual re-simulation (delete lifetime deposits and replay) or a residence/speed threshold sweep, rather than more seeds on the same null recovery tests.")
log(f"\n- {now}: {summ} Committed to the public repo (README section, figures/exp7).\n  -{next_note}")
g = lambda *a: subprocess.run(["git", "-c", "user.name=EV Campbell", "-c", "user.email=63378631+enactedvolition@users.noreply.github.com", *a], cwd=REPO, capture_output=True, text=True)
g("add", "README.md", "NIGHTLY_LOG.md", "mvp", "figures/exp7")
c = g("commit", "-m", "Exp 7: dark-room / niche-construction classification on Exp 5 pe_cost forks (RQ3 re-run)")
p = g("push", "origin", "main")
if p.returncode != 0:
    log(f"  - push failed: {p.stderr.strip()[:300]}")
    sys.exit(1)
print("committed and pushed", c.stdout[:300], p.stdout[:200])
