# Epistemology: what would count as agency here, and how we would know

Written 2026-11-28 (Phase 1 of EV's "recursively fix the whole thing from experiment 0"). Nothing was rerun to write this. Every number quoted below comes from `NIGHTLY_LOG.md`, `README.md`, `ARCHITECTURE.md`, `figures/exp*/` or the code, at commit `2ad06fd`. The per-experiment verdicts are in [`AUDIT.md`](AUDIT.md), and the ordered fix is in [`REBUILD_PLAN.md`](REBUILD_PLAN.md).

## 0. The short version

1. **The standing decision rule is the wrong yardstick.** "Active (closed-loop) recovery beats passive recovery with a CI excluding 0" is **not sufficient** for agency: a hand-implanted threshold rule passed it (Exp 20), and a thermostat would too whenever its feedback is worth something. It is also **not necessary** even for plain feedback: real feedback controllers fail it (Exp 27: a bang-bang thermostat was detected in 1 of 5 replicates; an ES-trained agent that provably uses energy was detected in 0 of 5). It mixes three different questions: does the system *use* the signal, does using it *help*, and is that use *goal-directed*. Each needs its own test.
2. **Most nulls in this repo are "no information", not "absence".** A null only counts as evidence of absence when (a) the metric has been shown to detect effects of the size in question in the same world and regime, (b) the test was pre-registered with an equivalence bound, and (c) the world actually rewards the capacity. Before Exp 27, no agency metric had met (a). Exp 22–24 also fail (c): Exp 25 showed energy feedback barely pays without a trade-off.
3. **Constructed systems are evidence about instruments, never about evolved populations.** The implant (Exp 19/20), COND (Exp 25/26), the P-ladder and ES (Exp 27) tell us what the assay can see. They say nothing about whether evolution produces agency.
4. **Validity checks must cover every manipulated arm.** The FROZEN one-step lead (Exp 25/26) went unnoticed because `clamp0 == ref` checks only the OPEN arm. Only Exp 21 ever tested "FROZEN with no drop == ref". A matching null-manipulation identity check is cheap, and it would have caught the bug before any data.
5. **Claims about bugs need the same scrutiny as claims about results.** The Exp 27 log says the Exp 22–26 evolved-population FROZEN arms "use the same lead". The code says otherwise: Exp 20–24 and `exp26_pop.py` feed the reference's *pre-step* energy (see [`AUDIT.md`](AUDIT.md) §B1). That sentence is wrong, and it had already been repeated in summaries.

## 1. What "agency" could operationally mean in this sim

Here is what the sim actually contains. Particles have a fixed-weight 16-unit Elman RNN (weights change only at birth, by mutation), an energy store E, a sensed copy of E (channel 8, `log1p(E) − 0.7`), and 8 actions (thrust ×2, eat, rep, share/attack, signal, bond, build). There is no within-life learning. So within one lifetime the only "adaptation" available is state-dependent control: the hidden state and the senses select the action. Policy change happens across generations, through selection on genomes.

This matters straight away. Any definition that requires the *policy* to change when the world's response changes can only be met by lineages, not by individuals. The project has been testing individuals with individual-level assays (the perturbation-recovery family) and lineages with lineage-level assays (Price, mixed competition) without saying which notion of agency each one targets. From now on, every claim names the level.

Below, each candidate definition gets: what it says, how it would be operationalised here, **what observation discriminates it from non-agency**, and what has gone wrong or could go wrong.

### 1.1 Graded control-theoretic levels (the backbone)

These are nested. Each level presupposes the one before it.

| level | name | operational test in this sim | discriminating observation | non-agency look-alike to rule out |
|---|---|---|---|---|
| **L0** | **causal sensitivity** to an internal variable | intervene on the *sensed* variable only (do(sensed E = x), body unchanged) and measure the change in action, over the first few steps before the world diverges | signed action response that is larger than the response of **unselected brains** (random or row-8-shuffled weights) | a random RNN also responds to its inputs: Exp 15 found evolved \|∂eat/∂E\| ≈ 0.047, the same as weight-shuffled brains (0.048). Non-zero sensitivity alone means nothing. |
| **L1** | **homeostatic (negative) feedback** | sign of the L0 response opposes the deviation (eat ↑ when sensed E ↓; withhold rep when E low), **and** the regulated variable returns towards its unperturbed course faster than with the signal frozen | CL beats FROZEN (signal-blind) on the regulated variable; OPEN (replay) is **not** the right comparator for this question | passive restoring physics: sharing gives 5% of E and theft takes 8% of the victim's E (`sim6.py:283–286`), the E_cap clip, and E-proportional costs. These pull E back with no controller at all. Only a signal-blind arm (FROZEN, or an input-ablated twin) separates them. |
| **L2** | **adaptive (valuable) feedback** | the feedback improves a viability or fitness outcome relative to the **best signal-blind policy** (fair search over blind policies, Exp 25-style), and **input ablation** of the agent itself costs fitness (Exp 27's U) | U = S(intact) − S(signal ablated) > 0 with pre-registered margin; holds against the best blind policy, not just a weak one | a sham fed a *harmful* signal (another agent's E) makes any rule look good (Exp 25: D_sham stayed +0.192 at haz = 0, where feedback does not pay). Blind comparators must be strong, not just blind. |
| **L3** | **goal-directedness / equifinality** (Rosenblueth, Wiener & Bigelow 1943; Ashby 1952) | the same goal state (e.g. E back in the viable band) is reached from **varied** perturbations through **different** action sequences, and the system compensates for perturbations it has never met (for example, a drop arriving through a new channel, or a moved food source) | persistence (goal reached despite obstacles) plus plasticity (different routes), beyond what a fixed stimulus–response table fitted to unperturbed behaviour predicts | a reactive map. Exp 24's GC (energy-blind gradient follower) recovered as well as the closed loop, so a simple reactive policy can look "equifinal" under one perturbation type. L3 needs several perturbation types, with GC-style reactive baselines refitted for each. |
| **L4** | **policy adaptation to changed contingencies** (Kenton et al. 2023, "Discovering agents", *Artif. Intell.* 322) | change the payoff structure (haz on/off, lean season, cost of eating) and check whether the **policy** shifts to suit it | across generations: the evolved policy moves in the direction the new contingency rewards, beyond drift (a shuffled-heredity control) | for fixed-weight individuals this is impossible within a life. COND, a fixed rule, *loses* at haz = 0 (Exp 25: −0.029) because it cannot adapt. That makes COND a device in Kenton's sense, not an agent. Evolution is the candidate agent at L4. |

Where the project stands: **L0 has never been tested properly with a selected-vs-unselected null on any evolved population.** Exp 15's prior was close, but it had no positive control. **L1 and L2 were tested with metrics that Exp 27 showed are insensitive.** L3 and L4 have never been tested. The only positive results are a constructed rule at L1/L2 (Exp 20) and lineage-level reward for an implanted rule (Exp 19, a selection result rather than an agency result).

### 1.2 Counterfactual sensitivity

**Says:** a system is agent-like with respect to a variable if its behaviour would have been different had that variable been different. This is L0 framed as interventions: the FROZEN arm is a counterfactual on the sense.

**Discriminates by:** an interventional contrast (CL vs FROZEN) measured on **actions**, close to the intervention, compared against a null distribution from unselected brains.

**Traps hit here:** (a) measuring the counterfactual's effect on a *distal outcome* (survival at t = 150) instead of on the action. The outcome mixes "uses the signal" with "using the signal pays here". Exp 25 shows the gap: COND's eat effort over steps 0–30 was CL 0.84 vs FROZEN 0.64, a huge and obvious behavioural difference, yet r_E and survival read it as a null. (b) Getting the counterfactual's timing wrong: the FROZEN post-step lead. (c) Sensitivity that is real but meaningless, because random weights also produce it.

### 1.3 Homeostatic feedback (cybernetic view; Wiener 1948, Ashby 1952)

**Says:** an agent keeps essential variables inside viability limits by negative feedback through action.

**Discriminates by:** L1. The regulated variable must be named in advance (energy here, never hidden-state distance; see §3.1). Restoration must exceed both physics-only restoration (signal-blind arm) and the noise floor.

**Trap:** a thermostat satisfies this completely. Homeostatic feedback is the *floor* of agency, not the ceiling. Passing L1 earns the words "feedback controller", not "agent".

### 1.4 The intentional stance (Dennett 1987; formalised as agents-vs-devices by Orseau, McGill & Legg 2018, arXiv:1805.12387)

**Says:** something is an agent to the extent that predicting it as a goal-seeker beats predicting it as a mechanism.

**Operationalise:** fit two model families to held-out trajectories: (i) a *device* model, a reactive map from current senses to actions (GC/GCX-style, with lags), and (ii) a *goal* model, a utility or viability-maximising policy with an inferred internal goal (e.g. a set point for E, solved by short-horizon planning in a learned forward model). Compare out-of-sample log-likelihood with a complexity penalty, **especially on novel perturbations**.

**Discriminates by:** the goal model predicting behaviour under perturbations not seen during fitting, where the device model fails.

**Traps:** it is stance-relative (a thermostat is well described by the stance); model-class choice is a forking path; fitting on the same trajectories you test is leakage. This is an L3 tool, worth building only once L1/L2 are positive.

### 1.5 Free-energy principle / active inference (Friston 2010, *Nat. Rev. Neurosci.* 11:127)

**Says:** persisting systems act to minimise (expected) surprise under a generative model; agency is action that reduces expected free energy, with pragmatic (preference) and epistemic (information-seeking) parts.

**Honest problem:** at the level of "anything that persists minimises free energy" the principle cannot be falsified. It discriminates nothing here unless a specific generative model is named.

**What was actually measured (Exp 2, 4, 5, 7):** surprisal under an **observer-side NLMS readout** that "is not inherited and does not affect behaviour" (`ARCHITECTURE.md` §13). That is the observer's model's surprise, not the agent's. The Exp 4 `pe_sense` predictor is likewise an add-on whose output is fed in as an input, not part of the brain's dynamics. So the dark-room results are about how predictable the environment is to an external linear model. They are not a test of active inference by the agents.

**Discriminating version, if wanted:** (a) the pragmatic part collapses to L1/L2 with viability as the prior preference; (b) the distinctive prediction is **epistemic action**: choosing actions that reduce uncertainty when uncertainty is costly, even when this has no immediate energy payoff. A test needs a world where information pays (a hidden food switch that can be resolved by sampling) and a comparator that does not sample. Not built; not a priority.

### 1.6 Causal emergence (Hoel, Albantakis & Tononi 2013, *PNAS* 110:19790; Rosas et al. 2020, *PLoS Comput. Biol.* 16:e1008289)

**Says:** a macro description (e.g. "this agent regulates E") can have more causal power or synergy than its micro parts. Ψ = I(V_t; V_{t+1}) − Σ_j I(X_j,t; V_{t+1}) > 0 is a sufficient criterion for emergence of the macro variable V over the micro parts X_j.

**Operationalise:** V = the agent's E (or position plus E); X_j = hidden units and senses. Estimate Ψ against (i) time-shuffled surrogates, (ii) the same agents with the energy channel ablated, and (iii) agent-free or blind-controller worlds.

**Discriminates by:** Ψ > 0 beyond all three nulls. Even then, emergence ≠ agency: storms and Gray–Scott spots (the sim10 line) have emergent macro variables. Causal emergence is at most a marker of *individuality*, which is a prerequisite for agency rather than agency itself.

**Traps:** macro-variable choice is a forking path; Gaussian MI estimators with ~100–200 samples (as in v2's Krakauer proxy) are biased; autocorrelation inflates I(V_t; V_{t+1}). An estimator ladder is needed first: systems with known Ψ.

### 1.7 Autonomy / normativity (Barandiaran, Di Paolo & Rohde 2009, *Adaptive Behavior* 17:367)

**Says:** an agent is (i) an individual, (ii) the source of modulation of its own coupling with the environment (interactional asymmetry), and (iii) modulating it in a way that serves its own viability (normativity).

**Maps onto:** (i) individuality (§1.6; Krakauer et al. 2020), (ii) L0, (iii) L1/L2. This is the cleanest summary of what the sim could support at the individual level, and it is the frame used in the rebuild.

### 1.8 Empowerment (Klyubin, Polani & Nehaniv 2005)

v2 used a lower-bound proxy I(A_t; X_{t+10} | X_t). It is a property of the *agent–world channel* and can be high for a wall-bumping random walker in a structured world. Without a ladder (known high- and low-empowerment controllers in this world), its value is uninterpretable. That is why v2's RQ2 is marked UNINFORMATIVE in the audit.

### 1.9 Which definition to adopt

No single definition. Claims are made **per level**, with the level in the claim: "evolved population X shows L0 sensitivity to E beyond the unselected null", "L2: input ablation costs Y survival". The word **agency** without a level is retired from this repo. If it is used at all, it means **L2 or higher plus individuality**, reached through the full ladder in §2.

## 2. The ladder of evidence

A claim may only be stated at the rung it reached. Rungs are cumulative.

| rung | what is required | what it licenses | where the project was |
|---|---|---|---|
| **R0 instrument integrity** | every off-path is bit-for-bit equal to the pre-change sim; **every** manipulated arm has a null-manipulation identity test (clamp0 for OPEN, **frozen0** for FROZEN, gc-identity for GC, sham-identity where definable); timing contracts written down (which step's state each arm sees); the checks run on real forks | "the code does what the pre-registration says" | strong for off-paths and OPEN; **missing for FROZEN in Exp 22–27** (only Exp 21 had frozen0); this is how the Exp 25/26 lead survived |
| **R1 metric validity on known controls** | a ladder of constructed positive controls of graded strength, negative controls that *can* fail, and shams that are neutral rather than harmful, all **in the same world and regime as the target**; output a sensitivity (power) curve, FPR with CI, and a minimum detectable effect (MDE) at the planned n | "the metric can see effects ≥ MDE here" | first done in Exp 27 (predation world only); not done for r_E, hidden-state distance, MI proxies, the self-built classifier, or Price-based selection detection |
| **R2 power for the target** | smallest effect of interest stated in calibrated units ("a P(0.5) servo's worth"); n chosen for ≥ 80% detection of it; clusters counted correctly (independent forks or evolutionary runs, not targets or replicates) | "a null will mean something" | never done before data; n was set by compute budget |
| **R3 pre-registration** | one primary contrast; decision rule including **equivalence bounds** (TOST; Lakens 2017) for claiming absence; analysis code committed (hash) before data; seeds used for any design choice excluded from the confirmatory set | "the result was not chosen after seeing it" | good habit since Exp 12; no equivalence bounds ever; design seeds reused (Exp 12 pooled discovery seeds; θ, haz and ES training all on the same 9 forks as the tests) |
| **R4 the test on the unknown system** | run exactly as registered; report everything, including the primary whatever it shows | a scoped result at the calibrated sensitivity | done routinely |
| **R5 robustness** | pre-declared secondaries: other drop sizes and horizons, other worlds (haz on/off), **independent evolutionary lineages** (new seeds, not new targets in the same 9 forks), mechanism checks (input ablation, lesion), negative-control populations (blind-evolved) | "not an artifact of one setting" | partial (horizons, drop sizes); since Exp 19 every result rests on the same 9 Exp 17 seed lineages |
| **R6 replication** | fresh lineages, fresh keys, and ideally a second implementation of the arm logic, or a code review by someone who did not write it | "not a bug or a fluke" | none |
| **R7 claim** | stated at the level reached (L0–L4), with what it does not show | — | — |

Two corollaries:
- **Absence claims** need R1 + R2 + R3 with an equivalence bound. A CI that includes the effect size of a known positive control is "no information", whatever its centre.
- **Validation of an instrument is itself a claim** and needs its own R0/R3. Exp 26's "the survival yardstick re-detects COND" was a validation claim, and it failed R0 (no frozen0 check), which is why the absence claims built on it collapsed.

## 3. Failure modes: hit, and could hit

Each entry names an instance in this repo. Experiment numbers refer to [`AUDIT.md`](AUDIT.md).

### 3.1 Unvalidated metrics
- **Hidden-state distance, closed vs open loop (Exp 3, 4, 5, 14).** Structurally biased against the closed loop. Open loop replays the reference inputs, so the only divergence is the kick inside a contracting map. Closed loop lets the kicked agent's changed actions change its inputs. Any brain whose actions depend on its hidden state, in any world sensitive to actions, will show "closed loop amplifies". A perfect energy regulator would show it too. The metric measures sensorimotor coupling times world sensitivity, not regulation. Exp 14's −19.25 is also bounded by float32 (open loop contracts to log r ≈ −21).
- **Gaussian MI proxies (Exp 2):** individuality and empowerment, never scored on a known agent.
- **r_E own-energy recovery (Exp 6, 14, 17, 22–24):** first checked on a known controller in Exp 25, where it missed.
- **Survival at t = 150 (Exp 26):** its "validation" was the lead artifact. Exp 27 then found it insensitive.
- **Self-built classifier and CF (Exp 7–13):** exact accounting (Exp 9), but no planted self-builder was ever detected.
- **Price/competition selection detection (Exp 11–19):** the identity is exact by construction (residual ~1e-16 is accounting, not evidence; methods audit 2026-11-06). Power to detect a given selection coefficient was never measured.

### 3.2 Nulls from insensitive tests read as absence
- Exp 6: "passive and closed-loop energy recovery are indistinguishable" at n = 3, with B CI [−1.299, +1.419] on a metric whose plausible effects are ≲ 0.3.
- Exp 14 / Exp 15's motivation: "evolved brains do not use the energy sense" (B ≈ 0). Unvalidated metric.
- Exp 22–24: "no active foraging recovery". Same metric, and in a world where (Exp 25) energy feedback has little to buy.
- Exp 26: "the three nights of nulls were not a metric artifact … CIs exclude effects a third the size of COND's". Withdrawn in Exp 27.
- **Rule:** before reading a null, ask "would this test have flagged a known feedback controller of the strength I care about, here, at this n?" If that has not been measured, the null is silent.

### 3.3 Leakage and timing bugs
- **FROZEN post-step lead (Exp 25 post hoc K = 200, Exp 26 COND):** `exp25_assay.py:47` / `exp26_cond.py:49` build `e_ref = ref["E"]`, the reference's energy *after* step t, and feed it at step t. A bang-bang rule fed the consequence of its own action is destabilised, so FROZEN looks bad and B inflates (+0.072 → +0.019 when corrected, Exp 27).
- **Post-step E in a correlation (Exp 17):** rep–E correlation computed with E after a birth (−0.79 E) is biased negative. Found in Exp 19.
- **Death-zeroed actions (Exp 6, 20):** the replayed action is zero on the step the reference dies. Caught. The fix (targets whose reference survives) conditions on survival, which excludes exactly the at-risk agents where regulation matters (methods audit §5).
- **Design/test overlap:** θ = 0.40 and haz = 0.004 were set from the Exp 20 forks; ES was trained on those same 9 forks (different keys); the assays test on them. That is fine for exploration, but it is not a held-out test.
- **Caught before damage:** the Exp 12 glob bug, and XLA 1-ulp re-fusion (Exp 9, 11, 15).
- **Claims about bugs can be wrong too:** the Exp 27 log's statement that Exp 22–26 evolved-population FROZEN arms carry the lead is false (they shift by one step correctly: `exp22_run.py:55`, `exp24_run.py:79`, `exp26_pop.py:71`, also `exp20_run.py:61`, `exp21_run.py:47`).
- **Could hit next:** an OPEN replay that clamps rep while the brain's rep head still drives other observer fields; GC maps fitted on `xprev` if `xprev` semantics ever change (today `xprev` = the input used at step t, written at `sim13.py:522`); `hu`/key reuse that makes "independent" replicates share hazard draws.

### 3.4 Garden of forking paths and multiplicity (Gelman & Loken 2013)
- Many contrasts per night, all as uncorrected 95% CIs, with "a lower bound just above 0 is called weak" as the only guard. Exp 23: two channels and three populations; Exp 26: three populations; Exp 11: about 6 contrasts × several measures at n = 3.
- **Post hoc rescues that then became the pre-registration:** Exp 25's survival read-out with K = 200 was post hoc, became Exp 26's primary, and was "confirmed" on the same 9 forks with the same buggy arm.
- **Rules changed after peeking:** Exp 6's active criterion was tightened mid-run (stricter, but still data-dependent). In Exp 16 the gate wording was changed before data, and that was documented, which is the right way to do it.
- **Discovery data reused as confirmation:** Exp 12's n = 9 primary includes the 3 seeds that produced the Exp 11 lead. The 6 new seeds alone give +0.167 [−0.032, +0.366].
- **Program-level forking:** each night's design is chosen after the previous night's result. That is fine for exploration. It means confirmatory claims need data that no design decision has touched.

### 3.5 Constructed or synthetic results treated as evidence
- The project never used fabricated data, and that discipline holds. The subtler failure is **constructed systems standing in for the question**. Exp 20's "the pre-registered decision rule is met" was achieved by implanting a threshold rule. It tells us the births-on assay can see a strong reproduction-timing feedback. It tells us nothing about evolved agency, and it shows the rule can be met by a device.
- Exp 26 used a constructed controller to validate an instrument, then used the "validated" instrument to support an absence claim about evolved populations. When the validation failed, the absence claim failed with it. Constructed results should feed into *calibration curves*, never into headline claims.

### 3.6 Definition drift
- "Uses the energy sense" has meant three things: non-zero ∂action/∂E (Exp 15 prior), selected for (Exp 15b, 17), and improves recovery (Exp 14, 22, 26).
- "Passive" switched between OPEN replay (Exp 6, 14, 20–24), offset-blind sensing (Exp 6, 14), FROZEN-to-reference (Exp 20+) and GC (Exp 24, 26). Each answers a different question. OPEN is **not specific** to feedback: stale replay is harmful by itself. Exp 27 blind controls show A-only false positives in 5/30 replicates, and P0 has A +0.008 [+0.003, +0.012].
- "Recovery" moved from hidden-state distance (Exp 3–5) to r_E (Exp 6) to lineage energy (Exp 20) to survival (Exp 26). Each change was reasonable, and none was validated before use.
- "Agency" stayed attached to the decision rule while the rule's meaning changed underneath it.

### 3.7 Goodharting the metric
- Once "active beats passive" became the bar, designs began to target the bar: implant a rule that will pass (Exp 19/20), build worlds around a hand-coded policy that pays (Exp 16–18, 25). That is legitimate as engineering a positive control. It becomes Goodhart when a pass is reported in the language of the original question ("agency claim: the rule is met …", Exp 20 summary).
- Pay-checks with a fixed hand policy can be gamed the same way: a world tuned until COND wins proves the world can reward COND, not that energy feedback is generally valuable.

### 3.8 Underpowered n
- Exp 1–11: n = 3 seeds, df = 2, t = 4.30. Nearly every "no effect" there is "not detectable at n = 3" (the repo's own §14 caveat).
- Exp 20–24: K = 40 targets per world; Exp 25 showed the survival read-out at K = 40 underpowered even for COND.
- Exp 27's detection rule: 5 replicates gives rates in steps of 0.2, and per-replicate detection discards the pooled information. Pooled B_surv for P1 is +0.010 [+0.005, +0.015], with a lower bound above 0, while P1 was "detected" in only 1/5 replicates. Part of "insensitive" is the rule, not the data.

### 3.9 Correlated replicates (pseudoreplication; Hurlbert 1984, *Ecol. Monogr.* 54:187)
- Targets in one world share the world. With 200 simultaneous targets out of ~300–400 eligible (Exp 26/27), they compete with each other.
- Exp 27's five "independent replicates" share the same 9 forks. The real n is 9 (correctly used for pooled CIs).
- **Every experiment from Exp 19 to Exp 27 descends from the same 9 Exp 17 `sw_pays` seed lineages.** Exp 21 even pairs across experiments by seed. Nine lineages is a small world sample. "9/9 seeds" across five experiments is not 45 confirmations.

### 3.10 Others found in this audit
- **Floor and ceiling of the effect.** The drop assay can only show recovery up to the drop's cost. In Exp 27 the 50% drop costs D = 0.035–0.062 survival, so B can never be large. Exp 26 pre-registered "if D's CI includes 0 the metric is uninformative in that population", and then reported COND as a PASS with D +0.024 [−0.008, +0.056]. **Its own rule was not applied.**
- **Value is non-monotone in feedback strength.** In the predation world, eating more after a drop also raises predation exposure, so B_surv peaks at P1–P2 and falls back for P4–P16 (Exp 27). A *presence* test must not use a score whose response to feedback strength is non-monotone.
- **Harmful shams.** SHAM_STEP and SHAM_P4 have B significantly *negative* (−0.021, −0.033). Replaying another agent's energy is a perturbation, not a null. "FPR 0/30" is also generous: 4 of the 6 blind controls have B ≡ 0 by construction, so only 10 replicates could have produced a false B.
- **Baseline asymmetry.** Exp 24's r*_GC is measured against GC0, a separate chaotic trajectory, while CL's baseline is ref. The contrast carries extra baseline noise on one side.
- **Process.** "Validity exact" meant that the arms the check covers are exact. Summaries generalised it to the whole assay.

## 4. What counts as a negative result worth publishing, and what is no information

### A negative result is informative when all of these hold
1. **The capacity pays in the tested world** (pay-check passed, with the best blind policy as comparator). Otherwise absence is the expected outcome and says nothing about the agents.
2. **The metric is calibrated** (R1) in that same world and regime, with an MDE.
3. **The test was pre-registered** with an equivalence bound δ in calibrated units, and the 90% CI lies inside ±δ (TOST). For example: "evolved B is within ±(half the B of a P(0.5) servo)".
4. **Several independent lineages** (fresh evolutionary runs, not more targets).
5. **The instrument passed R0** for every arm.

Then the claim is "in world W, at level L, evolved populations show less than δ, where δ corresponds to controller C". That is publishable and useful.

### Methodological negatives are publishable now
- Exp 27: a family of perturbation-recovery assays scored on a distal outcome is specific but insensitive. It missed a thermostat and an ES agent proven by ablation to use energy, and its sensitivity is capped by the perturbation's cost.
- The FROZEN timing artifact: a one-step lead in a frozen-input control fabricates a positive for threshold controllers (+0.019 → +0.072).
- Closed- vs open-loop hidden-state distance is structurally biased and cannot test regulation.
- A "closed beats passive" decision rule is met by a hand-built threshold rule, so it is not an agency criterion.

### No information
- Any null on an unvalidated metric: Exp 3–6, 14, 17 B, 22–24 (B), 2 (RQ2).
- Any null whose CI includes a known positive control's effect: Exp 26 populations vs STEP's B +0.020.
- Any null in a world where the capacity does not pay: Exp 22–24, 26 populations (no predation).
- Any null at n = 3 on a noisy metric: most of Exp 1–11.
- Any positive that rests on a constructed system, as a claim about evolved ones: Exp 20.

## 5. Standing rules from now on (proposed for `DECISION_PROTOCOL`-level adoption)

1. Claims name their level (L0–L4) and rung (R0–R7). "Agency" without a level is not used.
2. Every manipulated arm ships with its null-manipulation identity test on a real fork, in `tests/`, before any data. FROZEN replays the **recorded sensed input** (`xprev` channel 8, or the controller's recorded `u`), never a reconstruction from state.
3. No metric is used on an unknown system before it has a sensitivity curve on a known-control ladder in the same world.
4. Absence claims require TOST against a pre-registered δ in calibrated units.
5. Confirmatory runs use lineages and keys that no design choice has touched. Each pre-registration lists which seeds and forks informed the design.
6. Constructed systems appear only in calibration sections, never in headline results.
7. Claims about bugs cite `file:line`, and are checked by a second read of the code before they go into the log.
8. Effects are reported as estimates with CIs, plus, where relevant, "≈ k × the effect of controller C". Pass/fail language is reserved for the pre-registered primary.

## References (as cited above)
Ashby, W. R. (1952) *Design for a Brain*. · Barandiaran, X., Di Paolo, E. & Rohde, M. (2009) *Adaptive Behavior* 17:367–386. · Dennett, D. (1987) *The Intentional Stance*. · Friston, K. (2010) *Nat. Rev. Neurosci.* 11:127–138. · Gelman, A. & Loken, E. (2013) "The garden of forking paths" (working paper). · Hoel, E., Albantakis, L. & Tononi, G. (2013) *PNAS* 110:19790–19795. · Hurlbert, S. (1984) *Ecol. Monogr.* 54:187–211. · Kenton, Z. et al. (2023) "Discovering agents", *Artif. Intell.* 322. · Klyubin, A., Polani, D. & Nehaniv, C. (2005) IEEE CEC. · Krakauer, D. et al. (2020) *Theory Biosci.* 139:209. · Lakens, D. (2017) *Soc. Psychol. Personal. Sci.* 8:355–362. · McNamara, J. & Houston, A. (1986) *Am. Nat.* 127:358–378. · Orseau, L., McGill, S. & Legg, S. (2018) arXiv:1805.12387. · Rosas, F. et al. (2020) *PLoS Comput. Biol.* 16:e1008289. · Rosenblueth, A., Wiener, N. & Bigelow, J. (1943) *Philos. Sci.* 10:18–24. · Wiener, N. (1948) *Cybernetics*.
