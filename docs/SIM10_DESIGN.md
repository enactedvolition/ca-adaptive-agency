# sim10 — field-mediated "microbial ecology" world (design + pre-specified detectors)

Written **before** any tuning run (see git history: this file is committed before `runs10` exists).
Request (EV, 2026-11-11): richer per-agent parameters, a shared read/write N×N multi-channel field that
interaction goes through, tuned toward the organisational range of immune-system microbiology
(virus-like → eukaryote-like). Categories are **not** hard-coded; every entity runs the same rules with a
different genome. Detectors below are fixed in advance; results are reported against them, nulls included.

sim7–9 and the Exp 11/12 nightly pipeline are untouched (`mvp/sim10.py` is a new numpy-only file).

## World
* Torus, G×G cells (default G=48). Agents sit on integer cells; many per cell. Cap Nmax agents.
* **Field** `F[c, G, G]`, read/write by every agent, 5-point diffusion (per-channel D) + decay each step:
  | ch | name | source | sink / effect |
  |---|---|---|---|
  | 0 | `nut` nutrient | external inflow on patchy capacity map | uptake (competitive, per-cell) |
  | 1 | `org` organics/lysate | dead/lysed agents release biomass | uptake by agents with `eat_org` |
  | 2 | `tox` toxin | secreted (`tox_sec`) | damage ∝ tox·(1−`tox_res`) |
  | 3 | `sig` signal | secreted (`sig_sec`) | sensed: drives phenotype switch (`p`) |
  | 4 | `agm` antigen mass | every agent sheds its tag; lysis dumps tag | sensed by memory |
  | 5..12 | `agt[k]` antigen tag ×8 | tag-weighted shed mass (`agt/agm` = mean local antigen tag) | memory update |
* **Particles**: there is no separate virus class. Any agent can be small, non-motile, low-machinery; free
  agents diffuse/move cell-to-cell through the field grid and meet hosts only in a shared cell. Antigen
  (shed tags) is a field quantity.

## Heritable genome (35 numbers per agent, all in [0,1])
| gene | meaning | cost / trade-off |
|---|---|---|
| `size` | compartment/membrane size | maintenance ∝ (0.05+size)^0.75; uptake capacity ∝ size; engulf needs size ratio ≥1.5; infect needs ratio ≤0.5 |
| `auto` | own replication machinery | maintenance ∝ auto; free replication probability ∝ auto |
| `upt` | nutrient uptake rate | cost ∝ upt |
| `eat_org` | organics uptake rate | cost ∝ eat_org |
| `tox_sec` | toxin secretion | energy cost; harms neighbours (and self unless resistant) |
| `tox_res` | toxin resistance | maintenance cost |
| `sig_sec` | signal secretion | energy cost |
| `motil` | motility | move cost per move |
| `chemo` | chemotaxis up the nutrient gradient (vs random step) | — |
| `adh` | adhesion | cost; adhesive co-located agents (adh>0.5) move less and share energy |
| `share` | energy-sharing rate inside an adhesive cell group | gives energy away (cheating possible) |
| `engulf` | phagocytic drive (eat a smaller free agent in the cell) | cost ∝ engulf |
| `infect` | binding/entry drive into a larger agent | cost ∝ infect |
| `vir` | virulence: drain rate of host energy while inside | — |
| `mut` | mutation rate: σ = 10^(−3+2·mut) | none (indirect) |
| `defense` | non-self response: block entry / clear internal parasites / refuse engulfment-by... | maintenance cost |
| `mem` | memory learning rate (adaptive immunity) | maintenance cost |
| `p_thr`, `p_gain` | phenotype switch p = σ(gain·20·(sig_local−thr)); gain≈0 ⇒ no plasticity | — |
| `tag[8]` | surface tag (what others see) | — |
| `rec[8]` | receptor (what this agent binds to when infecting) | — |

Non-heritable state: energy E, age, host (−1 = free), memory vector m[8] and strength ms (copied to
offspring, CRISPR-like), expressed phenotype p.

Recognition: affinity(a,b)=exp(−|a−b|²/w), w=0.5 (8-d tags). Self-match = affinity(tag_i, tag_j).
* Infection of host h by free i (same cell, size_i ≤ 0.5·size_h): P = infect_i·aff(rec_i, tag_h)·(1−block_h),
  block_h = defense_h·(1−aff(tag_i,tag_h)) + ms_h·aff(tag_i, m_h)   (clipped to [0,1]).
* Inside: parasite takes vir·0.05·E_h per step; no field uptake. Replicates when E ≥ threshold with
  probability max(auto_p, auto_h) (**host machinery hijack**). Host clears each internal parasite with
  prob 0.1·block_h. Host death (E≤0) frees all internal parasites into the cell (burst) and lyses biomass to `org`.
* Engulf of free j by i (size_i ≥ 1.5·size_j): P = engulf_i·(1−aff(tag_i,tag_j)) (self-like is tolerated);
  i gains 0.6·E_j + biomass; j's internal parasites die.
* Memory: on clearing a parasite, and from local antigen whose mean tag is non-self, m moves toward that tag at
  rate mem, ms rises by mem·0.1 (decays 0.001/step).
* Phenotype p ("soma-ness"): uptake ×(0.5+p), free replication ×(1−p). With sharing in an adhesive group, a
  germ/soma split can pay; without it, it can't.


## Addendum A (EV t99u, 2026-11-11 13:39 PST, still before any tuning run): reaction–diffusion + release–reuptake

**Reaction–diffusion.** Channel 0 `nut` (U) and a new channel `V` (autocatalytic metabolite) follow
Gray–Scott kinetics in every cell, independent of agents:
  dU = D_U∇²U − U V² + f_gs·(K(x) − U)·inflow,   dV = D_V∇²V + U V² − (f_gs + k_gs) V
with documented constants D_U=0.16, D_V=0.08 (per step, 5-point), f_gs=0.035, k_gs=0.060 (spot/labyrinth
regime). So the field self-organises with no agents (checked by an agent-free run, reported as a sanity
check, not a result). V is food (uptake via transporters below). Agent-catalysed reactions (heritable
catalytic rates, acting on the agent's own cell):
  * `detox`: degrades toxin, d tox = −k_cat·detox·tox (k_cat=0.2), cost ∝ detox — neutralises others' toxin.
  * `digest`: exoenzyme converting organics to nutrient, org → U at rate k_cat·digest·org, cost ∝ digest;
    the product is a public good in the cell (cheats can use it).

**Release–reuptake.** Separate from passive nutrient uptake, each agent has heritable transporters:
`rt_vmax`, `rt_km`, `rt_spec`. Active Michaelis–Menten uptake per channel:
  J_sig = rt_vmax·rt_spec·sig/(Km+sig),   J_V = rt_vmax·(1−rt_spec)·V/(Km+V),   Km = 0.01+0.5·rt_km,
competitively split per cell when demand exceeds the cell stock. Reuptaken signal recovers 0.8 of its release
cost (recycling); V uptake is energy. Release is `sig_sec` (heritable). Transporter cost ∝ rt_vmax. Anyone with
the transporter can take up anyone's signal, so cheats can recycle others' release or shorten its lifetime.
Genome is now 40 numbers (24 scalars + tag[8] + rec[8]); field is 14 channels (U, V, org, tox, sig, agm, agt×8).

**D7 reuptake shapes signal lifetime/range (counterfactual).** At the final state of each run, clone it and
run 200 steps twice with identical RNG: intact vs reuptake knockout (rt_vmax→0 for the signal term only).
Measure (i) mean sig, (ii) effective signal half-life = ln2 / (removal rate), removal = decay + transporter
flux, from the intact run's flux accounting vs KO, and (iii) spatial correlation length of sig (distance where
the radial autocorrelation falls below 1/e). Fires if mean sig or correlation length differs by ≥ 20% KO vs
intact in the same direction in ≥ 3 seeds.

## Pre-specified detectors (fixed before tuning)
Genotype clusters: every W=250 steps, cluster living agents' 35-gene vectors (average-linkage, Euclidean cut
d=0.9 on a ≤1500 subsample; rest assigned to nearest centroid). Abundant cluster = ≥2% of agents.

* **D1 obligate parasitism.** Per window, per abundant cluster: rate_free = births while free / agent-steps
  free; rate_in = births while inside a host / agent-steps inside. Fires for a cluster-window if
  agent-steps inside ≥ 200, rate_in > 0, and rate_free ≤ 0.1·rate_in (incl. 0) while that cluster also has
  ≥ 200 free agent-steps (it does exist free, it just doesn't replicate there). Seed-level: fires if it
  holds in ≥ 3 windows in the second half of the run.
* **D2 host–parasite arms race.** Hosts_t = tags of agents infected during window t; Paras_t = receptors of
  agents that infected during t. A[t_p,t_h] = mean aff(rec, tag) over sampled pairs. Time-shift curve
  A(d)=mean over t of A[t, t+d]. Fires if (a) the infected-host tag centroid moves ≥ 0.3 (Euclid, 8-d) between
  the first and last infection windows and (b) A(0) − A(+8 windows) ≥ 0.05 (parasites are better on current
  hosts than on future ones = hosts escape). Needs infection in ≥ 12 windows.
* **D3 aggregation with division of labour.** Aggregate = cell with ≥ 4 agents having adh > 0.5.
  (a) fraction of agents in aggregates ≥ 0.10 averaged over the second half; (b) within-aggregate SD of p
  ≥ 0.2 on average (differentiated states), and (c) within aggregates, births concentrate in low-p members:
  share of births by p<0.5 members ≥ 0.7 while p≥0.5 members are ≥ 25% of members. Fires if (a)&(b)&(c).
* **D4 trophic depth.** Per window, energy flows: nutrient/organics → cluster (level 1 inputs), engulf and
  parasite drain flows cluster→cluster. Trophic level TL_k = 1 + Σ_j f_jk·TL_j / Σ f_·k (field income has TL 0).
  Report max TL among abundant clusters; fires if ≥ 2.5 (a consumer of consumers) in ≥ 3 second-half windows.
* **D5 diversity / persistence.** Effective number of clusters exp(H) per window; persistence = number of
  centroid-tracks (match d<0.9 between consecutive windows) lasting ≥ 1000 steps. Fires if median second-half
  exp(H) ≥ 3 and ≥ 3 tracks persist ≥ 1000 steps.
* **D6 immune memory.** Mean `mem` gene (second half) in `full` vs the `mem_inert` control (memory has no
  effect, same cost) — fires if full − inert > 0.05 with the paired-by-seed 95% CI excluding 0 (n≥3).

Controls (arms): `full`; `no_hijack` (inside replication uses only parasite's own `auto`: obligate parasitism
should become impossible — negative control for D1); `mem_inert` (for D6).
Sweep (robustness, not cherry-picking): nutrient inflow × {0.5, 1, 2} on `full`, seeds 0–2, plus the two
controls at inflow 1. All claims need ≥ 3 seeds; per-seed results are reported.

Not claimed in advance: anything is "immune-system-like" only to the extent the detectors fire. Phenomena
that are emergent here vs built in: the *mechanisms* (binding by tag–receptor affinity, host machinery use,
engulfment, adhesion, memory update, phenotype switch) are built in as possibilities; which genomes use
them, and whether parasites/aggregates/arms races appear, is not.

## Addendum B (EV t101u "Ensure thermodynamics is strict", 13:41 PST — built into the core before any tuning/sweep)

The whole step function was rewritten around two exact ledgers and a second-law check. This supersedes the
non-conservative parts of the text above (e.g. "E += income", clipping fields at 0, Gray–Scott as irreversible).

**Conserved quantities.** Matter (one element, "units") and energy. kT = 1 (isothermal bath).
* Matter lives in field species A, V, W, D (detritus), X (toxin), S (signal), Ag (antigen) and agent bodies M_i.
  Antigen *tags* (agt[8]) are information carried in proportion to Ag matter, not extra matter.
* Energy = Σ_s e_s·n_s (field) + Σ_i (E_i + e_B·M_i) (agents: ATP-like store E_i plus biomass). Heat goes to the bath.
* Boundary (the only exchange with outside): chemostat. Inflow of A at f·A_res per cell per step; dilution
  outflow f·c of **every** field species (agents are attached, not washed out). f = 0.035·`inflow`.
* Ledger asserted every step, run aborts on violation (`ThermoError`):
  M(t+1) = M(t) + in_M − out_M,  E(t+1) = E(t) + in_E − out_E − Q_bath,  |residual|/total < 1e-9.
  Q_bath is accumulated **explicitly** from each process's heat term (not computed by difference), so the
  check is meaningful. Founders (N0 agents, M0=0.3, E0=0.8 each) are part of the initial state.

**Free-energy table** (standard energies, kT units): e_A=10, e_V=6, e_W=0 (waste, ground state), e_D=5,
e_X=6, e_S=6, e_Ag=5, biomass e_B=12. Actual ΔG uses ideal-dilute activities (ΔG = ΔG° + ln Q).

**Field reactions (all reversible, mass action, k+/k− = exp(−ΔG°)):**
* R1 A + 2V ⇌ 3V, k+=1, k−=e^−4 (Gray–Scott autocatalysis with its reverse step). Catalysed by agent gene
  `cat_gs` (k±·(1+2Σcat_gs) in the cell: rate up, equilibrium unchanged).
* V, D, X, S, Ag ⇌ W with k+ = 0.06, 0.01, 0.02, 0.05, 0.02 and k− = k+·exp(−(e_s−e_W)). X⇌W catalysed by `detox`.
* With dilution f=0.035 and k_V=0.06 this is the (f,k)=(0.035,0.060) Gray–Scott regime, now reversible; patterns
  are maintained only by the boundary flux. Net fluxes are stock-limited (scaling keeps the sign), so no
  negative concentrations and dissipation stays ≥ 0. Diffusion is conservative explicit 5-point (4D < 1).
* Entropy production of each reaction: σ = (J+−J−)·ln(J+/J−) ≥ 0.

**Agent processes, each with heat and dissipation (= T·entropy production, asserted ≥ 0 per process):**
* Catabolism A→W (passive uptake, competitive per cell): only when ΔG_actual = e_W−e_A+ln(W/A) < 0; the agent
  captures η=0.5 of −ΔG_actual into E, the rest is dissipated. Same for detritus D→W (`eat_org`).
* Assimilation A→biomass (grows M up to 2·M_target): paid from E at (e_B − e_A − ln A) + 0.2 overhead (uphill,
  driven by coupling to the E store; dissipation = overhead).
* Active MM transporters (`rt_vmax`, `rt_km`, `rt_spec` split between S and V): pay transport work
  max(0, ln(c_int/c_out)) + 0.1 overhead per unit (c_int = 1), then catabolise inside (capture η·(−ΔG°+ln(W/c_int)));
  uptake that would not pay is returned unchanged (no uphill transport without coupling).
* Release (`sig_sec`, `tox_sec`) and shedding draw **body matter** (B→S, B→X, B→Ag, downhill; heat e_B−e_s).
* Toxin hit: X + E → W (+ heat), agent loses E.
* Maintenance (size, body mass, every costly gene), movement (∝ 0.2+M), division (c_div), engulf attempt,
  energy sharing (10% loss), parasite drain (20% of transferred E lost): E → heat. Nobody can pay with E it lacks;
  an agent that cannot pay maintenance and is at E≈0 dies.
* Reproduction splits M and E exactly in half (offspring gets real matter/energy; mutation is free and creates
  nothing). Parasite offspring are made from matter and energy drained from the host — "virions carry real mass
  built by the host". Engulf: predator gets 0.6 of prey M and E, rest → detritus + heat.
* Death: body → detritus (90%) + antigen (10%, carrying tags) in the cell; leftover E → heat. Never deleted.
* Approximation (stated): release from body treats the body as a pure phase (activity 1), and the release heat
  e_B−e_s is counted as fully dissipated; for very high local S this slightly overstates dissipation, never
  understates it below 0 for the tested ranges (asserted every step).

**Required sanity tests:** (i) closed system (`inflow=0`): no boundary flux; must relax toward equilibrium
(A, V → ~0, matter ends in W/D) and all agents die; (ii) agent-free run (`N0=0`): reversible Gray–Scott must
still self-organise under the boundary flux. Both run as part of the sweep (`closed_s*`, `agentfree_s*`).

**Sweep knob change (documented honestly):** the agent-free check (13:40 PST) showed Gray–Scott dies if the
feed is spatially patchy, so feed is uniform and the sweep knob `inflow` now scales the chemostat rate f
(0.5, 1, 2), not a patch map. inflow=2 moves Gray–Scott to f=0.07 (a different, likely homogeneous, regime).

## Addendum C — post-design changes for the v2 economics run (EV t102u "Go + compute analytics", 2026-11-11 14:39 PST)

These are **parameter/mechanism changes made after seeing the v1 sweep**. The seven detectors (D1–D7) and their
thresholds are **not changed**. Thermodynamic accounting is unchanged and still asserted every step. All v2
settings live in `mvp/sim10.py: V2` (used via `preset=2`); v1 defaults are untouched, so v1 is reproducible.

| change | v1 | v2 | why (v1 diagnosis) |
|---|---|---|---|
| `c_auto` maintenance per unit own machinery | 0.004 | 0.02 (5×) | losing machinery didn't pay → no obligate parasites |
| replication energy threshold | E > c_div+0.1 | E > c_div+0.1+0.5·M (`E_rep_M`) | tie replication to body mass |
| growth target used by assimilation | 0.2+size | 0.05+0.5·size (same as division target) | v1 grew to a target different from the division target |
| founder `size` | U(0,1) | U(size_lo, size_hi), default (0,1); sweep small (0,.35)/full/large (.65,1) | test whether "founders too big" blocked infection |
| founder body mass | 0.3 for all | 2·(0.05+0.5·size) (`M0_by_size`) — taken from the initial state, counted in the ledger | small founders really are small |
| internal transporter concentration `c_int` | 1.0 | 0.01 | transport work ln(c_int/c_out) kept; v1 value made reuptake cost ≈4.6 kT > its return |
| transporter overhead `ovh_tr` | 0.1 | 0.02 | same |
| signal release cap `sig_cap` | 0.004 | 0.01 | raise local signal |
| transporter Vmax scale `k_rt` | = k_upt (0.06) | 0.02 (separate knob) | V uptake by transporters drained the RD spots |
| passive uptake `k_upt` | 0.06 | 0.02 | slower drawdown of A |
| grid / caps | G=40, Nmax 5000, N0 1500 | G=64, Nmax 12000, N0 3000 | bigger world for patterns |
| initial V seeds | 12 squares | 12·(G/40)²·2 ≈ 61 squares (`v_seed_mul=2`) | at G=64 with 12 seeds the agent-free RD died (probe, 14:45) |
| trace autocatalyst in the feed `V_res` | 0 | 0.002 (a documented boundary input of V at f·V_res; counted in in_M/in_E) | lets RD spots re-nucleate after a crash; it is a boundary flux, not a source term |

Note on the c_int change: the total free energy of "signal outside → waste" is unchanged; c_int only sets how
it splits between transport work (ln(c_int/c_out) ≥ 0 paid up front when c_int > c_out) and the catabolic step
inside (−ΔG° + ln(W/c_int)). Dissipation ≥ 0 is still asserted per process.

Pre-run probes (14:45–14:52 PST, 3–4k steps, recorded here, not tuned to the detectors): agent-free RD keeps
spots in 3/3 seeds at G=64 (mean V ≈ 0.05–0.06); with agents, **every probed uptake setting (k_upt 0.01–0.03,
k_rt 0.01–0.06, with or without trace feed V) still lost the spots** after the first population boom
(t≈1000), while k_upt=0.01 starved the population to ~100. I did not find a setting that keeps both; v2 uses
k_upt=k_rt=0.02 and reports the spot loss as a result.

Runs: base, no_hijack, mem_inert × seeds 0–2 × 20k steps; agent-free (20k) and closed (8k) checks × 3 seeds;
robustness sweep c_auto {0.004, 0.02, 0.06} × founder size {small, full, large} × seeds 0–2 × 8k steps (base model).
New per-window analytics (not detectors): signal half-life ln2/(removal rate), V spot count (components of
V>0.1) and V variance, transporter net energy and work, entropy production and max ledger residual per window.
