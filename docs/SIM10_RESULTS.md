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
  mostly from engulfment chains. Caveat found after running: within-cluster feeding (cannibalism) is 4–39% of
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

---

# sim10 v2 — economics fix + analytics (2026-11-11, runs 14:42–14:56 PST)

Post-design changes are in `SIM10_DESIGN.md` Addendum C (5× machinery cost, mass-tied replication threshold, founder
size range with mass tied to size, payable reuptake via c_int 1→0.01 and overhead 0.1→0.02, signal cap 0.004→0.01,
G 40→64, slower uptake k_upt 0.06→0.02 and transporter k_rt 0.02, more initial V seeds, trace V in the feed).
Detectors D1–D7 unchanged (D1–D5 code is exec'd verbatim from `analyze_sim10.py`). 42 runs: base / no_hijack /
mem_inert × 3 seeds × 20k steps; agent-free (20k) and closed (8k) × 3; robustness sweep c_auto {0.004, 0.02, 0.06}
× founder size {small (0–.35), full (0–1), large (.65–1)} × 3 seeds × 8k steps. 3 procs, 9–151 s/run, max RSS 1.0 GB.
Full tables: `figures/sim10_v2/tables_v2.md`; per-run JSON `figures/sim10_v2/detectors_v2.json`.
CIs are 95% t-intervals across seeds (n=3, df=2 — wide).

## Thermodynamics (all 42 runs, every step asserted)
Max |matter residual| 7.7e-11, max |energy residual| 1.4e-10 (tolerance 1e-9). The residual grows ~linearly with
run length from float accumulation (≈7e-11 at 20k steps; `timeseries_v2.png`), so runs beyond ~200k steps would
need compensated summation (Kahan) to stay under 1e-9 — not relaxed here. Min per-process dissipation 0; per-step
entropy production > 0 everywhere (min 4.8e-4 in the dying closed systems). Closed systems: all agents dead by
step 500–750 in 3/3; A → 0.003–0.006, V → 0.0034 (= its equilibrium with W, e^−6·W), no spots. Agent-free:
13–18 self-sustaining V spots at 20k in 3/3 seeds (mean V 0.054–0.066). Pass.

## Detector results (unchanged definitions)
| condition (20k) | D1 | D2 | D3 | D4 | D5 | D7 |
|---|---|---|---|---|---|---|
| base | 2/3 | 0/3 | 0/3 | 3/3 | 3/3 | no |
| no_hijack (negative control for D1) | **3/3** | 0/3 | 0/3 | 3/3 | 3/3 | no |
| mem_inert | 2/3 | 0/3 | 0/3 | 3/3 | 3/3 | no |
D6: base − mem_inert `mem` gene, paired +0.084 [−0.229, +0.397] → no.

Key statistics, base (mean [95% CI], n=3): D2 host-tag drift 0.52 [0.26, 0.78] (criterion ≥0.3 met), A(0)−A(+8)
0.0083 [0.0033, 0.0133] (criterion ≥0.05 not met — 6× too small); D3 clump fraction 0.12 [−0.11, 0.35],
within-clump SD of p 0.030 [0.010, 0.049] (criterion ≥0.2); D4 max TL 3.04 [2.59, 3.49], without self-loops
2.85 [2.61, 3.09], self-loop share 0.11 [0.04, 0.17]; D5 exp(H) 9.9 [−3.9, 23.7]; signal half-life 7.9 [7.1, 8.6] steps.

### What D1 firing means (diagnosed, not claimed)
D1 fires as pre-specified in 2/3 base seeds, **but it also fires in 3/3 no-hijack seeds**, the negative control in
which a parasite can only replicate with its own machinery. So D1 isn't detecting machinery hijacking. The
clusters that meet it spend 1–4% of their time inside a host, and inside it they replicate ~10× faster because
they're feeding on host matter and energy. But **73–92% of their births still happen while free** (one seed: 25%).
They're facultative intracellular parasites, nutritionally favoured inside hosts, not obligate ones. Only base s0
looks virus-like in its genes: hit-cluster centroid auto 0.10 vs population 0.47, size 0.56 vs 0.91, infect 0.88
vs 0.44 — one seed, not a finding. Pooled over all agents, inside birth rate is ~2× free in every condition,
controls included. Under the substantive definition ("replication ≈ 0 without a host") obligate parasitism did
**not** emerge.

### D2 cross-time test
Cross-time curve A(d) = parasite receptor at t vs host tag at t+d, d in 250-step windows (`tables_v2.md`). Base s0
falls monotonically from past to future, 0.199 (d=−16) → 0.162 (d=+16): parasites are best on *past* hosts and
hosts move away, which is directional host escape. Seeds 1–2 peak at d=0 with a ≤0.01 drop. Controls look the
same, and so does no_hijack. The effect size is ~0.01 against a pre-set threshold of 0.05. No measured arms race;
at most weak, directional host-escape drift. `tag_receptor_drift_v2.png` shows host-tag and receptor PC1 over
time plus the A[t, t'] matrix.

### D3
Clumps form, with up to 20–36% of agents in clumps for thousands of steps in some seeds, but phenotype variance
inside them is ~0.03–0.05, an order of magnitude under the threshold. Births are 83% by low-p members, but that's
because almost all members are low-p. No differentiated states and no division of labour.

### D7 reuptake (signal half-life / range vs knockout)
Reuptake now runs and pays per molecule. Net capture is 0.18–0.70 kT per unit taken up after the
ln(c_int/c_out)+overhead work. Knockout lengthens the signal half-life in 9/9 main runs (intact 6.9–9.2 → KO
9.0–9.6 steps; KO/intact 1.23 [0.95, 1.52] base) and raises mean signal 1.04–1.44×. But the ≥20% same-direction
criterion is met in ≥3 seeds only in sweep cell c_auto 0.004 × full (sig KO/intact 1.36, 1.32, 1.21), not in base
or the controls. Spatial correlation length is unchanged within grid resolution (2–4 cells). **Reuptake doesn't pay
for its transporters:** estimated transporter maintenance over the second half (0.0015·rt_vmax·N·steps) is 15k–39k
energy units vs 350–7.4k of net capture, 10–60× more. Consistent with that, the `rt_vmax` gene sits near 0.5–0.6,
which is drift-like. (Caveat: Vmax is shared with V uptake, and V is ~0 after the spots die, so half the
transporter capacity is idle.)

### Robustness sweep (`robustness_v2.png`)
* Machinery cost does select: mean `auto` drops 0.53–0.59 (c_auto 0.004) → 0.32–0.42 (0.02) → 0.21–0.26 (0.06).
* D1 hits **fall** as machinery gets costlier (c_auto 0.004: 0–2/3 seeds; 0.02 and 0.06: 0/3 everywhere). Hosts
  shed machinery too, so there's less to hijack, and inside replication uses max(own, host) machinery.
* Founder size doesn't matter. Small founders (size 0–0.35) evolve back to large: mean size 0.62–0.83 by 8k steps,
  0.93 in full/large. Being big pays (uptake ∝ size, engulfing needs ≥1.5× mass, and infection needs the *host* to be
  ≥2× bigger, so selection on hosts and engulfers drives size up). "Founders too big" was not the blocker.
* D2 fires in 1/27 sweep runs (c_auto 0.004 × small, one seed). D3 0/27. D4 fires in 26/27, D5 in 25/27.
* RD spots are wiped out in 26/27 sweep runs and in all 9 main runs (spot count 0 after t≈1000).

## Organisational levels: emerged vs not (v2)
* **Emerged robustly:** free-living autotrophic cells; engulfers forming a trophic chain (TL ≈ 3 without self-loops,
  in all conditions); persistent genotype diversity (still partly founder standing variation); transient adhesive
  clumps; facultative intracellular parasites that use the host as food.
* **Did not emerge:** obligate (virus-like) parasites; a measured arms race (Red Queen); division of labour in
  clumps; adaptive immune memory (D6 null); agents exploiting or modulating RD patterns (spots die in every agent
  run); reuptake shaping signal range.
* **Why, in numbers:** (1) machinery hijack can't pay when hosts are themselves shedding machinery, and inside time
  is only 1–4%. (2) Size selection is uniformly upward, so a small-parasite / large-host split never stabilises.
  (3) Phenotype switching has no signal gradient to work on inside a clump (corr length 2–4 cells, half-life ~8
  steps), so there's no germ/soma split. (4) The population boom at t≈1000 draws A below the spot-sustaining level,
  and V never re-nucleates even with trace V in the feed.

## Recommended next step
1. **Decouple host and parasite machinery economics.** Make hijack use host machinery *capacity* while the host
   pays for it, and make machinery a prerequisite of free replication rather than a rate multiplier (threshold).
   Then low-auto genomes only replicate inside. This is still emergent: nobody is assigned the role.
2. **Break uniform size selection:** a surface/volume uptake law (uptake ∝ size^(2/3), maintenance ∝ size) so small
   bodies have a niche.
3. **Protect the RD layer:** a spatially separate feed for A vs a slow-diffusing V substrate, or an agent
   carrying-capacity term set by thermodynamics (e.g. finite A feed per cell). Verify spots survive with agents
   before running evolution.
4. Split the transporter Vmax per channel (separate genes) so signal reuptake isn't taxed by an idle V transporter.
5. Add Kahan-compensated ledgers before going past 100k steps.
6. Then 20k × 5 seeds for base / no_hijack / mem_inert, same detectors.
