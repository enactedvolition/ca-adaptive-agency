# Rebuild plan: ordered, recursive, foundations first

Written 2026-11-28 (Phase 1). Progress is tracked in the Status checklist below. Each step is gated on the one before it. Framework: [`EPISTEMOLOGY.md`](EPISTEMOLOGY.md). Verdicts being repaired: [`AUDIT.md`](AUDIT.md).

## Status (nightly routine: pick up at the FIRST unchecked step; a step is checked only when every sub-item is)

Numbering note: the 2026-11-28 morning instruction called the harness "step 1" and the signal-use test "step 2"; in this file they are **Step 0** and **Step 1**.

- [ ] **Step 0: instrument integrity**
  - [x] sim13 `ctl["e_fix_use"]/["e_fix"]` hook; off path == pre-change sim13 bit for bit (tests/test_assay_core.py)
  - [x] `mvp/assay_core.py` shared arms (ref, cl, open, sense-level frozen, ablate, fixown) + timing contract (module docstring)
  - [x] identity tests: e_fix absent, clamp0, frozen0, fixown (ablate0), controller clamp0/frozen0, deliberate post-step-lead failure caught
  - [x] `docs/PREREG_TEMPLATE.md`
  - [ ] gc-identity test (GC arm not yet in the harness)
  - [ ] Exp 27 regression through the harness == `report27.json` (needs a src=1 sham hook in the harness policy; ~65 min detached)
- [ ] Step 1: S0 signal-use metric (pre-registration + known-controller ladder validation)
- [ ] Step 2: input-ablation value V
- [ ] Step 3: fresh lineages
- [ ] Step 4: re-test evolved populations (TOST)
- [ ] Step 5: selection-detection calibration
- [ ] Step 6: kick assay retire/rebuild
- [ ] Step 7: births-on calibration
- [ ] Step 8: dark rooms / niche
- [ ] Step 9: v2 proxies
- [ ] Step 10: world where feedback pays
- [ ] Step 11: higher levels

## How "foundations first" is ordered

The order follows **dependency depth**, not experiment number. The deepest layer is the instrument: the sim plus the assay arms that every later claim runs through. Next come the metrics, which must be calibrated before they are used. Then the populations, which must be independent of design choices. Last come the re-tests of the old questions. Inside each layer, the agency line (energy feedback, L0–L2) goes first, because EV's question is about agency and because Exp 14–27 hang off it. The v2 proxies (Exp 1/2a) are the oldest experiments but carry the least weight, so they come late.

## Recursion rule (applies to every step)

1. A step **passes** only if its pre-registered gate passes. If it fails, stop, fix that step's metric or design, and re-run *that step's* calibration. Nothing downstream runs on a failed instrument.
2. When a step passes, update the matching `AUDIT.md` rows (new verdict, commit hash) and re-propagate through the dependency graph. A claim moves from UNINFORMATIVE to STANDS or WITHDRAWN only through a pre-registered test at the rung it claims.
3. Every pre-registration lists (a) the code hash, (b) **which seeds and forks informed any design choice** (these are excluded from confirmatory tests), (c) one primary contrast, (d) a TOST equivalence bound δ in calibrated units for any absence claim, and (e) the null-manipulation identity test for every arm.
4. Constructed systems appear only in calibration outputs. Headline results are about evolved populations, at a named level (L0–L4).

**Compute unit.** A "night" means one routine session (~45–60 min of attended work) plus detached compute on this box (8 CPU cores, ~3.5 GB free RAM on bad nights, ≤ 4 procs for sim13 jobs). Reference timings from the log: an Exp 27 ladder assay ≈ 5 s per system per seed (12 systems × 5 reps × 9 seeds plus two ES trainings plus a pay-check ≈ 65 min wall); evolution of 3 arms × 9 seeds × 3000 steps ≈ 15 min (Exp 17); a 1500-step implant run ≈ 14 min (Exp 19); a 10k-step run of 9 jobs ≈ 35 min (Exp 13); a K = 200 population assay ≈ 7 min for 27 jobs (Exp 26).

## Overview

| step | layer | repairs | metric | must pass | nights |
|---|---|---|---|---|---|
| **0** | instrument | all assays (B1, B3, the Exp 25/26 lead) | — (integrity) | identity tests for every arm; Exp 27 numbers reproduced through the new harness | 0.5 |
| **1** | metric | L0 everywhere (Exp 14, 15 prior, 22, 26b) | interventional action sensitivity S0 | P-ladder monotone; ES and implant detected; FPR ≤ 5% vs unselected-brain null | 1 |
| **2** | metric | L2 everywhere (Exp 6, 14, 17B, 22–27) | input-ablation value V (+ calibrated drop assay as a secondary) | g_min defined; ES detected ≥ 80%; power curve in both worlds | 1–2 |
| **3** | populations | single fork family (all of Exp 19–27) | — | fresh lineages reproduce the Exp 17/19 recipes without tuning | 1 |
| **4** | re-test | Exp 14, 17B, 22, 24, 26b | S0, V from steps 1–2 | pre-registered TOST on old + fresh lineages | 1 |
| **5** | metric | Exp 11–19 selection claims (B5) | Price ΣS power; fair blind-policy search | planted-trait power curve; blind-timer comparators | 1–2 |
| **6** | retire/rebuild | Exp 3b, 4, 5, 14 kicks (B6) | — | pre-registered retirement, or a constructed hidden-state regulator detected | 0–1 |
| **7** | metric + re-test | Exp 20/21 births-on assay | births-on drop assay, calibrated | rep-timing ladder; E-matched control | 1 |
| **8** | metric + re-test | Exp 2b, 7–13 dark rooms / niche | classifier + true CF | planted self-builders detected | 1–2 |
| **9** | metric or retire | Exp 1, 2a | λ estimator, MI proxies | planted σ² networks; ladder systems | 1 |
| **10** | new science | — (the logged next step) | S0, V, Price | blind-timer-inclusive pay-check, then evolution on fresh seeds | 3 |
| **11** | higher levels | — | equifinality, contingency shift, Ψ, stance | only if step 10 gives an L2 positive | 3+ |

Total ≈ 14–18 nights, if each gate passes the first time. Expect re-dos.

## Step 0: instrument integrity (R0)

- **What:** one shared module (`mvp/assay_core.py`) for every perturbation arm, replacing the copy-pasted `run_one` bodies in `exp20`–`exp27`.
  - **FROZEN replays the recorded sensed input** (the reference's `xprev[:, 8]`, or a controller's recorded `u`), never a value reconstructed from E. That makes the post-step-lead class of bug impossible by construction.
  - A new additive sim13 hook `ctl["e_fix"]` (per-agent exact value written into channel 8; absent means the code path is unchanged) replaces the `e_off = target − E` trick, which leaks ULP-level information (noted in Exp 27).
  - A written **timing contract** for each arm: which step's state it reads, and which step's action it writes.
- **Identity tests (pytest, real forks, bit-for-bit):** clamp0 (OPEN, no drop) == ref; **frozen0** (FROZEN, no drop) == ref; ablate0 (e_fix to the agent's own current sensed value) == ref; gc-identity (GC with a map that reproduces the reference thrust exactly, built from recorded actions) == ref; `e_fix` absent == pre-change sim13 on all fields. A test that **fails on purpose** when FROZEN is fed post-step E, so the harness is shown to catch the Exp 25/26 bug.
- **Regression:** Exp 27's pre-registered table recomputed through the new harness must match `report27.json` exactly (FROZEN there is already sense-level). This is a re-execution of existing code for a regression check, not a new experiment, and it happens in Phase 2.
- **Pre-registration template** (`docs/PREREG_TEMPLATE.md`): code hash, design-seed ledger, primary, δ, identity tests.
- **Compute:** tests only, minutes. **0.5 night.**

## Step 1: L0 metric, interventional action sensitivity (presence of signal use)

- **Metric S0.** For each target, a **pure sensor intervention** with the body untouched: do(sensed E ← sensed value of E·(1 − d)) for d ∈ {0.25, 0.5} via `e_fix`, against a sham intervention (do(sensed E ← its own value), which by step 0 equals ref). Read out the change in each action head (eat, rep, thrust, social) over steps 1, 3 and 5 after onset, **before world divergence matters**. That is a direct, monotone measure of ∂action/∂sensed-E. Also report its sign relative to homeostasis (eat ↑ / rep ↓ when sensed E ↓). The primary is the signed eat response at k = 3, pooled over targets, with seeds (forks) as the unit.
- **Null distribution, the key negative control:** the same S0 computed on **unselected brains**. These are (i) the same evolved weights with the energy row (W_ih row 8) shuffled across agents (Exp 15's construction), and (ii) random-initialisation brains in the same world state. Detection means "the signed S0 exceeds the 95th percentile of the null". This answers the trap that random RNNs respond to inputs too (Exp 15: \|es\| 0.047 ≈ shuffled 0.048).
- **Ladder:** P(g) for g ∈ {0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 16}; STEP; ES (**retrained on held-out forks**, see below); the Exp 19 implant in its RNN form (a known L0/L1 positive inside a real brain); `blind = 1` brains (S0 ≡ 0; integrity only); SHAM-STEP (S0 should be ≈ 0 to own E); constant controllers.
- **Worlds:** haz = 0 and haz = 0.004; brains sighted and blind (four cells).
- **Gate (pre-registered):** S0 is monotone in g (Spearman over the ladder, CI > 0); P(0.25) detected in ≥ 80% of 9-seed replicates; ES and the implant detected; FPR ≤ 5% (Clopper–Pearson upper bound reported) over shuffled-row and random-brain nulls; MDE in G_eff units stated.
- **ES retraining:** train on forks/keys disjoint from all test forks. The Exp 20 forks have 9 seeds, so train on Exp 22 SHAM forks or on fresh step-3 lineages, or split 9 → 4 train / 5 test, which is stated as a limitation.
- **Compute:** S0 needs ≤ 10 steps per run. Ladder × 4 cells × 9 seeds × 5 reps plus ES retraining (~30 min) **≈ 1 night.**

## Step 2: L2 metric, input-ablation value (does using the signal help?)

- **Primary metric V** (generalises Exp 27's U, which detected ES where the drop assay could not): within-world parity design with common random numbers (Exp 25). A random half of the cohort has its sensed E fixed by `e_fix` to a constant (its population's median sensed value at the fork, so no information remains), the other half intact; then swap. V = S(intact) − S(ablated), read at T = 150 and T = 300. A positive V means the signal pays in this world for this controller.
- **Secondary, the drop assay recalibrated:** survival with the corrected FROZEN arm, an 80% drop (Exp 26: D 0.09–0.11, informative), and **pooled-seed detection** instead of per-replicate. This is kept only if it beats V's power on the ladder.
- **Ladder:** as step 1, plus the best blind policies (constant efforts, blind timers) as negatives, and **neutral shams** (another agent's E time-shifted to match own-E statistics, or a recorded own-E trace from another run) instead of the harmful random-partner sham.
- **Worlds:** haz = 0 (**the world the Exp 22–26 populations actually lived in**) and haz = 0.004; sighted brains, because that is what the evolved populations have.
- **Gate:** power curve V(g) with detection probability vs G_eff; g_min defined (the smallest gain with ≥ 80% detection, where all larger gains also qualify); ES detected ≥ 80%; FPR ≤ 5% on blind policies and neutral shams. Output **δ for TOST** in step 4: half of V for P(g_min), in survival units.
- **If V fails the gate in the haz = 0 world:** record that "energy-feedback value is not measurable where the evolved populations live", and drop all absence claims about them for good. That is itself an informative negative about the *world*.
- **Compute:** about 12 systems × 2 worlds × 2 parities × 9 seeds × 5 reps of 300 steps, detached **≈ 1–2 nights.**

## Step 3: fresh, independent populations

- **Why:** every population result in Exp 19–27 descends from the same 9 Exp 17 `sw_pays` lineages. Those lineages also informed θ, haz and ES training.
- **What:** re-run the **Exp 17 recipe** (sim13 rep_free = 1, rep_fixed = 1; arms sw_full / sw_pays / sw_pays_blind; 3000 steps) and the **Exp 19 implant recipe** (TRUE/SHAM, 1500 steps) on **new seeds 100–117** that no design choice has touched, with forks saved. No parameter changes, and nothing re-tuned.
- **Pre-registration:** the recipes verbatim, plus a descriptive replication check: Exp 17's low blind rep rate (~1.5%) and Exp 19's fixation (Δshare TRUE > 0) reported with CIs on the fresh seeds. A failure to replicate is reported and stops the steps that depend on it.
- **Compute:** about 18 seeds × 3 arms × 3000 steps (~30 min) plus 18 × 2 × 1500 (~30 min) **≈ 1 night.**

## Step 4: re-test the evolved populations at L0 and L2

- **Repairs:** Exp 14 (B), the Exp 15 prior, Exp 17 (B), Exp 22, Exp 24, Exp 26b.
- **Metrics:** S0 (step 1) and V (step 2), as calibrated.
- **Populations:** old (Exp 17 sw_pays t = 3000, Exp 20 TRUE t = 1500, Exp 19 SHAM t = 1500, labelled replication) and fresh (step 3, confirmatory).
- **Pre-registration:** primary = V in fresh sw_pays at haz = 0; L0 primary = signed S0 vs the unselected-brain null; **TOST with δ from step 2** for any "no use" claim; three populations named in advance, and Holm correction across them.
- **Possible outcomes:** (i) S0 > null and V > 0: L1/L2 present (scoped to that world). (ii) S0 > null, V inside ±δ: the signal is used but does not pay at the calibrated resolution. (iii) S0 ≈ null (TOST): no L0 use beyond unselected brains, the first legitimate absence claim in the repo. (iv) Anything else: no information, reported as such.
- **Compute:** about 3 populations × 2 sets × 9–18 seeds, with assays of minutes each, **≈ 1 night.**

## Step 5: selection-detection calibration (the Exp 11–19 claims)

- **Metric:** exact Price ΣS on a marker trait, and mixed-competition Δshare.
- **Ladder:** a **planted neutral marker gene** (FPR), and planted markers with a known fitness effect s (a small per-step energy bonus tied to the marker) on a grid of s; heredity-shuffled controls. Output: power of ΣS at n = 9, 3000 steps, as a function of s.
- **Fair blind-policy search (B5):** add **blind refractory timers** (wait k steps after a birth or attempt, k on a grid) and memory-based blind policies to the Exp 18 gate and the Exp 25 pay-check comparator sets.
- **Re-assess:** Exp 12 on fresh seeds only (the discovery seeds are excluded); Exp 13 and 15b with stated MDE; Exp 18's gate against the best blind *policy*.
- **Compute:** planted-trait evolutions about 6 s-values × 9 seeds × 3000 steps (~45 min) plus timer gates (~20 min) plus Exp 12-type fresh runs (~30 min) **≈ 1–2 nights.**

## Step 6: hidden-state kick assay (Exp 3b, 4, 5, 14): retire or rebuild

- **Recommendation: retire** it as an agency test, pre-registered in one paragraph citing B6. Closed- vs open-loop hidden-state distance cannot separate regulation from sensorimotor coupling in a world that responds to actions.
- **If EV wants it kept:** construct a positive control, an RNN whose actions provably steer its own hidden-state norm back via its inputs, plus a matching device. Require the metric to separate them before rereading Exp 3–5 or 14.
- **Compute:** 0 nights (retire) or 1 night (rebuild).

## Step 7: births-on assay calibration (Exp 20/21)

- **Metric:** the Exp 20 lineage-energy assay through the step 0 harness, with pooled-seed detection.
- **Ladder:** reproduction-timing controllers of graded gain (soft thresholds around C), blind timers and Bernoulli rates as negatives, and the implant as the top rung.
- **Re-test Exp 21 specificity** against an **E-matched** control population (targets matched on E0 distribution, same target-selection rule), not just an age-matched one.
- **Compute:** ≈ 1 night.

## Step 8: dark rooms and niche construction (Exp 2b, 7–13)

- **Metric:** the exact self-built classifier (Exp 9) and true CF vs equal-mass control.
- **Ladder:** **planted self-builders** (hand-coded build-and-stay agents at graded build rates and residence), planted non-builders as negatives. Power of CF as a function of build rate and of the number of dark targets.
- **Then:** rerun the self-built and CF tests on fresh seeds only if the instrument detects planted builders at the build rates evolution actually produced. Otherwise record "uninformative at evolved rates".
- **Priority:** low (dark-rooming ≠ agency). **≈ 1–2 nights.**

## Step 9: v2 proxies and criticality (Exp 1, 2a)

- λ estimator on planted networks with known σ² (0.5–1.5); individuality, empowerment and heritability proxies scored on the step 1–2 ladder systems and on blind controls. Any proxy that does not separate known agents from devices is **retired** with a note.
- **≈ 1 night.**

## Step 10: the world where energy feedback pays (the logged next step, now properly gated)

- births-on sim13 + predation + lean season, plus a prediction-error energy cost as a second arm (EV's standing goal).
- **Pay-check first:** lineage competition, aware vs the **best blind policy including timers** (step 5), pre-registered margin.
- **Evolve** on fresh seeds (9+), with controls: no-predation world, shuffled-energy sham, sensor-blind arm. Exact Price with residual.
- **Assay** with S0 and V (calibrated **in this world**: rerun the steps 1–2 ladders here first), TOST, Holm.
- **≈ 3 nights** (pay-check 1, evolution 1, calibration + assay 1).

## Step 11: higher levels (only if step 10 gives an L2 positive)

- **L3 equifinality:** several perturbation types (an energy drop, a food-field shift, a movement handicap, a sensor delay), each with its own refitted reactive baseline (GC/GCX-style). Goal-state reattainment beyond the baselines.
- **L4 contingency shift:** switch haz on or off mid-evolution and test whether the evolved policy shifts in the rewarded direction beyond the shuffled-heredity drift (Kenton et al. 2023, at lineage level).
- **Intentional-stance model comparison** (goal model vs device model, out-of-sample on novel perturbations), and **causal emergence Ψ**. Each needs its own estimator ladder (systems with known Ψ; known agents vs devices) before use.
- **≈ 3+ nights.**

## First three steps, concretely

1. **Step 0:** `assay_core.py` with sense-level FROZEN replay, the `e_fix` hook, identity tests for every arm (including a test that fails on purpose on post-step input), the pre-registration template, and an Exp 27 regression. 0.5 night.
2. **Step 1:** the L0 interventional action-sensitivity metric S0, with an unselected-brain null, calibrated on the P-ladder, STEP, a held-out-trained ES and the RNN implant, in 4 world × brain cells. 1 night.
3. **Step 2:** the L2 input-ablation value metric V, calibrated on the same ladder plus blind timers and neutral shams, in haz = 0 and haz = 0.004, producing g_min, a power curve and the TOST δ. 1–2 nights.

Nothing is assayed on an evolved population until these three pass.
