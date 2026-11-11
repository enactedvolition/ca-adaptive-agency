# sim10 — first sweep results (2026-11-11, ~13:48–13:52 PST)

Sweep: `full` at inflow (chemostat rate multiplier) 0.5 / 1 / 2, `no_hijack`, `mem_inert` (inflow 1), the closed
system (`inflow=0`) and agent-free (`N0=0`) checks; seeds 0–2; 6000 steps (closed: 8000). 21 runs, 3 procs,
~15–22 s and ~535 MB max RSS per run. Raw outputs live outside the repo in `/workspace/sim10_runs/sweep`
(npz not committed). Per-run numbers: `figures/sim10/detectors.json` / `detectors_stdout.txt`.
Detectors are the ones fixed in `SIM10_DESIGN.md` before tuning. **No tuning was done after the sweep**: this is
the first parameter set, so "didn't emerge" here means "didn't emerge at these settings", not "can't".

## Thermodynamics (all 21 runs, every step asserted)
* Matter ledger |residual|/total ≤ 2.6e-13; energy ledger ≤ 9.7e-13 (tolerance 1e-9). No `ThermoError`.
* Minimum per-process dissipation = 0 (never negative); per-step entropy production > 0 in every step of every run.
* Closed system (3 seeds): all agents dead by step ~500–750; A falls to 2.5e-5, V to 0.003, matter ends as W
  (+ trace D/X/S/Ag at their reverse-reaction equilibria). Entropy production per step falls to ~4e-7. Pass.
* Agent-free (3 seeds): reversible Gray–Scott settles into 2–3 self-sustaining V spots under the boundary
  flux (mean V 0.011–0.033) — self-organised but sparse; the irreversible GS at the same (f,k) gave labyrinths
  (mean V ≈ 0.17) in the pre-check, so the reverse step + strict stock limits push it toward the spot regime.
* With agents, A is drawn down to ~0.05–0.1 and **V spots go extinct** in every seed: agents out-compete the
  autocatalyst, so agents do not end up exploiting or modulating RD patterns. Strictness did not cost speed
  (~2.5 ms/step at ~1000 agents); it did change the GS regime (above).

## Detectors (per seed: s0 / s1 / s2)
| detector | full in0.5 | full in1 | full in2 | no_hijack | mem_inert |
|---|---|---|---|---|---|
| D1 obligate parasitism | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ |
| D2 arms race | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ |
| D3 aggregation + division of labour | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ | ✗✗✗ |
| D4 trophic depth ≥2.5 (as pre-specified) | ✓✓✗ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| D5 diversity/persistence | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✗ |
| D7 reuptake KO changes sig/range ≥20% (same dir, ≥3 seeds) | ✗ | ✗ (s0 only: corr length ×1.75) | ✗ | ✗ | ✗ |
D6 memory: mean `mem` gene, full − mem_inert per seed −0.092, −0.069, −0.024; mean −0.062, 95% CI [−0.148, +0.024]. ✗.

Reading, bluntly:
* **D1 null.** Agents do get inside hosts (2–49 at a time out of ~700–2300) and drain them, but no abundant
  cluster replicates inside while failing to replicate free. The `no_hijack` negative control is therefore
  uninformative. Likely cause: `auto` is cheap enough (0.004/unit) and hosts are short-lived, so dropping machinery
  doesn't pay; and parasites start large (founders are uniform random), so few satisfy M ≤ 0.5·M_host.
* **D2 null.** Infected-host tags drift 0.29–0.62 (criterion ≥0.3 mostly met), but parasites aren't better on
  contemporary than future hosts by ≥0.05 (largest gap 0.043, full in0.5 s0; full in2 s0 0.037). Drift, not a
  measured Red Queen.
* **D3 null.** Adhesive aggregates exist in a few runs (18.7% of agents in full in0.5 s2; 36% in mem_inert s2) but
  phenotype SD inside them is 0.01–0.12 (< 0.2): no differentiated states, no germ/soma split.
* **D4 fires but is weak evidence.** Max trophic level ≥2.5 among abundant clusters in 14/15 non-control runs,
  mostly from engulfment chains. Caveat found after running: within-cluster feeding (cannibalism) is 5–39% of
  consumer flow and inflates TL (some windows hit 51 from self-loops). A post-hoc variant excluding self-loops still
  fires in 14/15 (median max TL 2.2–4.2). Post-hoc, so treat as descriptive.
* **D5 fires but is mostly standing variation.** Founders are uniform-random 40-d genomes, so many clusters exist
  from t=0 and persist under low mutation; exp(H) 2.6–22.7. This is not evidence of emergent niche partitioning.
* **D7 null.** Evolved transporters rarely take up signal at all: transport work ln(1/S) at S ~ 0.01 (≈4.6 kT +
  overhead) exceeds η·ΔG of catabolising it, so the strict thermodynamics makes signal reuptake unprofitable and
  it switches off. KO changes nothing in 14/15 runs.
* Nothing here is "comparable to immune-system microbiology". What emerged: free-living cells, engulfers eating
  other clusters (a trophic chain), transient intracellular parasites, occasional adhesive clumps, persistent
  genotype diversity. What didn't: obligate parasites, arms races, multicellular division of labour, immune memory,
  agents shaping RD patterns, reuptake-controlled signal lifetime.

## Recommended next run
Fix the obvious economics before more seeds, then rerun the same detectors unchanged (pre-registered):
1. Make the machinery trade-off real: raise `auto` cost ~5× and scale replication threshold with body mass so
   small, low-`auto` agents are cheap; seed founders with a size gradient (variety, not classes).
2. Make signal worth retrieving: lower transport overhead or raise local S (higher sig_sec cap / lower D_sig), so
   reuptake can pay and D7 becomes testable; keep the ln(c_in/c_out) term.
3. Protect RD from total drawdown (larger grid or slower uptake k_upt) so the V patterns survive to be exploited.
4. 20k steps × 3 seeds × {full, no_hijack, mem_inert}, ≤3 procs (~2 min/run), checkpointed; a size/auto sweep
   (3×3) to test robustness of whatever fires.
