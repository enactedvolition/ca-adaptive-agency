# ca-adaptive-agency

A JAX agent world with free-moving particles. Each particle has a small random recurrent brain (an Elman RNN: 16 sense inputs (15 in v1–v3), 16 hidden units, 8 actions), an energy budget, mutation, and an environment it changes: resources deplete and regrow, agents deposit walls and pheromone, and the weather is noisy.

The question is whether anything here earns the words **edge of chaos**, **agency**, **dark room**, or **self-regulation** once controls are in place. Every claim is gated against controls: shuffled heredity, no mutation, no energy cost, static environment, and inert walls. Statistics are 95% t-intervals over 3 seeds (df = 2, so the intervals are wide). Nulls are reported as nulls. All numbers come from the simulations in this repo; there is no synthetic data.

Design and full results: [`ARCHITECTURE.md`](ARCHITECTURE.md). Licence: Apache-2.0.

## Quickstart

You need Python 3.11–3.13. No GPU is needed.

| | one-liner |
|---|---|
| **Windows** (PowerShell) | `git clone https://github.com/enactedvolition/ca-adaptive-agency; cd ca-adaptive-agency; .\run.bat` |
| **Mac / Linux** | `git clone https://github.com/enactedvolition/ca-adaptive-agency && cd ca-adaptive-agency && ./setup.sh && ./run.sh` |

- `setup.sh` / `setup.ps1` create `.venv`, install the pinned `requirements.txt` (CPU JAX), and run the smoke test. The smoke test is a tiny world of 100 steps that takes a few seconds and writes `out_smoke/frame.png`.
- `run.sh` / `run.bat` run a short demo, about 30 s on CPU: 1024 agent slots for 1500 steps. It writes `out_demo/frame.png` and `out_demo/demo.gif` and opens the GIF. `run.bat` runs the setup first if it hasn't been done.
- Make targets: `make setup | smoke | run | test`.
- NVIDIA GPU (Linux or WSL2): `./setup.sh gpu`, which uses `requirements-gpu.txt`. Native Windows and macOS run on CPU.
- VS Code / Codespaces: open the folder in the dev container (`.devcontainer/`).
- Windows script error ("running scripts is disabled"): `run.bat` already bypasses it. For setup alone, run `powershell -ExecutionPolicy Bypass -File setup.ps1`.

## Results so far

| experiment | question | answer |
|---|---|---|
| v1 (`sim.py`) | Does anything agent-like beat controls? | MVP world and visualizations ([figures/v1](figures/v1)) |
| v2 (`sim2.py`, `exp2.py`) | Edge of chaos? Agency? Dark rooms? | **No criticality.** Energy cost drives brains into order (median λ_max −0.16 → −0.70). Agency metrics beat no control in 0/16 cells. Dark rooms appear only when depletion or cost is removed; walls cut surprisal by ~4 nats but aren't selected for ([figures/v2](figures/v2)). |
| Exp 3 (`exp3_price.py`, `exp3_perturb.py`) | Is the ordered regime held by individuals or by turnover? | **Turnover.** An exact Price decomposition shows selection dominating. Perturbations decay passively under open-loop and frozen inputs, but the closed behavioural loop *amplifies* kicks (closed − open log r(50) ≈ +13 to +16, CI > 0). This is population sorting into stable inherited dynamics, not individual regulation ([figures/exp3](figures/exp3)). |
| Exp 4 (`sim4.py`, `exp4_*.py`) | Make regulation pay: (a) agents sense their own prediction error, (b) instability costs energy | **No** (see below) |
| Exp 5 (`sim5.py`, `exp5_*.py`) | Make regulation pay by charging for prediction error in a switching world | **No.** Still passive recovery; no variant's closed loop beats replay (see below) |
| Exp 6 (`sim6.py`, `exp6_regulate.py`) | Score recovery on energy: closed loop vs a clamped passive body after a 50% energy drop | **No.** Passive and closed-loop energy recovery are indistinguishable or passive wins (see below) |
| Exp 7 (`exp7_darkroom.py`) | RQ3 re-run: do self-built dark rooms appear under pe_cost with depletion still on? | **No.** Self-built dark rooms do not appear under pe_cost while depletion is on (see below) |
| Exp 8 (`exp8_counterfactual.py`) | Threshold sweep + dynamic CF_own: is Exp7 self-built null fragile? Does CF reclass other-built? | **No — self-built null robust (tiny/zero even at loosest grid point).** CF_own reclass other→self (sw_no_cost frac≈0.052); see section |
| Exp 9 (`exp9_attr.py`, `exp9_cf.py`) | Exact lifetime deposit attribution: is the Exp 7/8 self-built zero a proxy artifact? | **Exact self-built @100/0.2 CI>0 in: none; true CF beats equal-mass control in: none.** sw_no_cost exact self-built +0.0000 [+0.0000, +0.0000]; see section |
| Exp 10 (`exp10_run.py`, `exp10_cf.py`) | Wall-protected intake: do privately valuable walls produce self-built dark rooms? | **Build up in: none; selection on build CI>0 in: none; self-built CI>0 in: none.** See section |
| Exp 11 (`exp11_run.py`, `exp11_cf.py`) | Mass-ownership guard dose-response: is building selected once its private return can beat its cost? | **Build up in: none; selection on build CI>0 in: none; self-built CI>0 in: none.** See section |
| Exp 12 (`exp12_run.py`, `analyze12.py`) | Pre-registered seed extension of Exp 11 (n = 9): is building selected more when heredity is intact? | **Primary ΣS(build) sw_mg2 − sw_mg2_shuffled CI>0 (+0.146 [+0.027, +0.264]); primary mean-build contrast not (+0.047 [−0.003, +0.097]).** See section |
| Exp 13 (`exp13_run.py`, `analyze13.py`) | Long horizon (10000 steps, n = 9): does the Exp 12 selection on building turn into more building? | **Pre-registered mean-build contrast sw_mg2 − sw_mg2_shuffled over t ≥ 5000 does not hold (+0.011 [−0.045, +0.068]).** Transient divergence at 2000–6000 steps, gone by 8000. See section |

## Experiment 4: make regulation pay

New `sim4.py` knobs:
- `pe_sense`: input 16 is the agent's own prediction error, log1p(10·MSE), from an online predictor that reads the agent's hidden state.
- `lam_cost`: an energy cost of 0.01 per unit of positive local log tangent-stretch, charged every step.

Seven variants × 3 seeds, N = 2048 slots, 3000 steps: full, shuffled, no_cost, static_env, pe, lamcost, pe_lamcost. Perturbations follow the Exp 3 protocol, from full-state forks at step 2500.

**Verdict: paying for regulation did not produce regulation.** Every variant, both arms included, lands in the same row of the decision rule: selection dominates and recovery is passive, which the rule labels **stable inherited dynamics**. No arm makes the closed loop beat passive replay.

- **Kick ε = 0.1, closed − open-loop log r(50)** (positive means the closed loop is worse):

  | variant | closed − open [95% CI] |
  |---|---|
  | full | +15.0 [+7.9, +22.1] |
  | pe | +17.5 [+9.7, +25.3] |
  | lamcost | +18.5 [+8.9, +28.1] |
  | pe_lamcost | +18.3 [+13.2, +23.5] |

  Open-loop and frozen replays contract kicks by e^-15 to e^-21. The closed loop stays at order ε.
- **Ongoing noise, log(closed/open) over steps 50–100:**

  | variant | log(closed/open) [95% CI] |
  |---|---|
  | full | +2.74 [+2.55, +2.94] |
  | pe | +3.12 [+2.25, +3.99] |
  | lamcost | +3.24 [+1.85, +4.62] |
  | pe_lamcost | +3.38 [+2.34, +4.42] |

- **Arms vs full, paired by seed:** no arm improves on full. lamcost is borderline *worse* on kicks: +3.45 [−0.03, +6.94]. Only no_cost and static_env, the starvation-free controls, are less amplifying.
- **Price decomposition, cumulative over 3000 steps:**
  - Under standardized inputs, S dominates and W ≈ 0 in every variant. For lamcost, S = −1.00 [−2.23, +0.22] and W = −0.008.
  - Under actual inputs, within-life drift W is negative in every variant except lamcost, where W = −0.28 [−0.79, +0.22] and S+T dominates in 2 of 3 seeds.
  - lamcost has the most ordered population (Δλ̄ = −1.01 [−2.02, −0.004]).
  - Identity residual: ≤ 1e-15 for trait (b); ≤ 7e-6 for trait (a), which is float32 accumulation.
- **Why the cost did nothing:**
  - Selection had already made the brains ordered (λ < 0), so a cost on *positive* stretch rarely triggers. Late per-step energy spend is 0.018–0.020 in both full and lamcost.
  - The PE input is sensed, but nothing selects for using it.
  - The extra recovery cost is |Δ| ≤ 0.017 energy over 100 steps.
- **Next:** charge the cost on prediction error itself rather than on λ > 0; use a changing task where fast re-adaptation pays; run more seeds and longer runs.

Figures: [figures/exp4](figures/exp4) (`price4_cumulative.png`, `lambda4_traj.png`, `perturb4_recovery.png`, `perturb4_arms.png`). Full tables: `figures/exp4/tables4.md`.

## Experiment 5: charge energy for prediction error, in a changing world

New `sim5.py` knobs (with both off, `sim5` reproduces `sim4` exactly):
- `pe_cost`: energy cost of 0.01 × the instantaneous mean squared error of the agent's own next-input predictor (which reads its hidden state), charged every step after a 10-step warm-up. In calibration it added about a quarter to a third to per-step spend.
- `sw_period`: the resource layout cycles through 4 patch layouts every 300 steps, so food moves and fast re-adaptation pays.

Seven variants × 3 seeds, N = 2048 slots, 3000 steps: sw_full, sw_pe, sw_pecost, sw_pecost_pe (main arm), pecost_pe_static (no switching), sw_pecost_pe_shuffled, sw_no_cost. Same Price and perturbation protocols as Exp 4 (forks at step 2500).

**Charging for prediction error did not produce regulation either.** No variant's closed loop beats passive (open-loop or frozen-input) replay, so the decision rule gives no agency label.

- **Kick ε = 0.1, closed − open-loop log r(50), and ongoing noise log(closed/open), steps 50–100** (positive = closed loop worse):

  | variant | closed − open [95% CI] | ongoing [95% CI] | decision-rule label |
  |---|---|---|---|
  | sw_full | +16.78 [+10.96, +22.60] | +3.19 [+1.89, +4.48] | stable inherited dynamics |
  | sw_pe | +16.90 [+13.43, +20.38] | +3.22 [+1.89, +4.55] | stable inherited dynamics |
  | sw_pecost | +19.18 [+14.16, +24.20] | +3.28 [+1.33, +5.23] | stable inherited dynamics |
  | sw_pecost_pe | +17.77 [+13.28, +22.27] | +3.28 [+2.01, +4.56] | stable inherited dynamics |
  | pecost_pe_static | +17.64 [+10.33, +24.95] | +3.06 [+1.06, +5.07] | stable inherited dynamics |
  | sw_pecost_pe_shuffled | +20.27 [+13.77, +26.78] | +3.59 [+2.47, +4.70] | stable inherited dynamics |
  | sw_no_cost | +11.23 [+10.50, +11.95] | +2.00 [+1.71, +2.28] | stable inherited dynamics |

- **Paired contrasts by seed** (negative = first arm regulates better):

  | contrast | cond | Δ(closed − open) |
  |---|---|---|
  | sw_pe vs sw_full | kick0.1 | +0.12 [-4.05, +4.28] |
  | sw_pe vs sw_full | ongoing | -0.07 [-1.36, +1.23] |
  | sw_pecost vs sw_full | kick0.1 | +2.40 [+0.85, +3.95] |
  | sw_pecost vs sw_full | ongoing | -0.29 [-1.07, +0.49] |
  | sw_pecost_pe vs sw_full | kick0.1 | +0.99 [-1.36, +3.34] |
  | sw_pecost_pe vs sw_full | ongoing | -0.09 [-0.92, +0.73] |
  | sw_pecost_pe_shuffled vs sw_full | kick0.1 | +3.49 [-6.37, +13.35] |
  | sw_pecost_pe_shuffled vs sw_full | ongoing | +0.21 [-0.19, +0.60] |
  | sw_no_cost vs sw_full | kick0.1 | -5.55 [-10.98, -0.13] |
  | sw_no_cost vs sw_full | ongoing | -0.65 [-2.65, +1.35] |
  | sw_pecost_pe vs pecost_pe_static | kick0.1 | +0.13 [-3.55, +3.81] |
  | sw_pecost_pe vs pecost_pe_static | ongoing | +0.17 [-1.05, +1.40] |
  | sw_pecost_pe vs sw_pecost_pe_shuffled | kick0.1 | -2.50 [-10.03, +5.03] |
  | sw_pecost_pe vs sw_pecost_pe_shuffled | ongoing | -0.30 [-1.24, +0.65] |
  | sw_pecost_pe vs sw_pe | kick0.1 | +0.87 [-1.01, +2.76] |
  | sw_pecost_pe vs sw_pe | ongoing | -0.03 [-0.51, +0.46] |

- Late prediction-error MSE per step: sw_full 0.738, sw_pecost_pe 0.614, pecost_pe_static 0.644, shuffled 0.713.
- Price identity residual: max 1.9e-15 (standardized inputs), 1.9e-06 (actual inputs).

Figures: [figures/exp5](figures/exp5). Full tables, including the Price decomposition and post-switch re-adaptation: `figures/exp5/tables5.md`. Generated automatically by `mvp/finalize5.py` from `report5.json`.

## Experiment 6: score recovery on regulated variables (energy), not hidden-state distance

Exp 3–5 always found the closed loop amplifying hidden-state kicks, because a perturbed agent moves differently and the coupled world diverges. Exp 6 asks whether agents restore the variables they would actually regulate. It reuses the Exp 5 forks (step 2500) and the same 40 matched targets per run, 7 variants × 3 seeds, 120 steps. New `sim6.py` is `sim5` plus an observer-side intervention hook (with no intervention it is `sim5` exactly).

- **Energy-drop test.** The target loses 50% of its energy at t = 0. Closed loop: it senses energy (input channel 8) and behaves freely. Passive control: same drop, but its actions are clamped to what its unperturbed twin did. Sensor-blind control: closed loop, but the energy sense reads as if nothing was lost. Recovery fraction r = 1 − mean deficit over steps 60–120 / amount removed. Active recovery means A = r_closed − r_clamped > 0 with the 95% CI excluding 0. Validation: clamping an unperturbed agent to its own actions reproduces the reference exactly (max |ΔE| 0.0e+00).
- **Hidden kicks on regulated variables.** Kicks of ε 0.1 and 0.3 are scored by |ΔE| against the unperturbed reference and compared with a world-chaos floor: the same agent in an unperturbed twin world whose weather/birth random key differs.

**No variant restores lost energy faster than a passive body replaying the unperturbed agent's actions**, so the decision rule still gives no agency label on the regulated variable either.

  | variant | r closed | r clamped (passive) | A = r_cl − r_clamp | B = r_cl − r_blind | energy recovery |
  |---|---|---|---|---|---|
  | sw_full | +0.746 [+0.514, +0.979] | +0.667 [-0.066, +1.401] | +0.079 [-0.831, +0.989] | +0.109 [-0.663, +0.881] | no detectable difference |
  | sw_pe | +0.558 [+0.120, +0.996] | +0.663 [+0.104, +1.223] | -0.106 [-0.377, +0.165] | -0.261 [-0.764, +0.242] | no detectable difference |
  | sw_pecost | +0.455 [-0.446, +1.356] | +0.765 [-0.132, +1.662] | -0.310 [-0.409, -0.211] | -0.065 [-0.540, +0.411] | passive ≥ active |
  | sw_pecost_pe | +0.543 [+0.029, +1.057] | +0.819 [+0.433, +1.205] | -0.276 [-0.430, -0.121] | +0.060 [-1.299, +1.419] | passive ≥ active |
  | pecost_pe_static | +0.352 [-0.775, +1.480] | +0.657 [+0.270, +1.044] | -0.304 [-1.222, +0.613] | -0.043 [-0.967, +0.881] | no detectable difference |
  | sw_pecost_pe_shuffled | +1.206 [-0.886, +3.298] | +0.899 [+0.080, +1.718] | +0.307 [-1.132, +1.746] | +0.434 [-0.469, +1.336] | no detectable difference |
  | sw_no_cost | +1.093 [+0.817, +1.368] | +0.967 [+0.765, +1.169] | +0.125 [+0.035, +0.216] | +0.115 [-0.108, +0.338] | no detectable difference |

  | variant | mean |ΔE| kick 0.1 | kick 0.3 | twin floor | log(kick 0.1 / floor) |
  |---|---|---|---|---|
  | sw_full | 0.152 | 0.191 | 0.291 | -2.315 [-5.966, +1.335] |
  | sw_pe | 0.160 | 0.190 | 0.281 | -1.773 [-5.089, +1.543] |
  | sw_pecost | 0.104 | 0.191 | 0.330 | -4.036 [-7.393, -0.678] |
  | sw_pecost_pe | 0.181 | 0.229 | 0.358 | -2.327 [-5.152, +0.498] |
  | pecost_pe_static | 0.090 | 0.203 | 0.364 | -3.369 [-5.394, -1.345] |
  | sw_pecost_pe_shuffled | 0.148 | 0.210 | 0.327 | -2.501 [-6.634, +1.633] |
  | sw_no_cost | 0.090 | 0.097 | 0.100 | -0.171 [-0.293, -0.048] |

Hidden kicks move energy more than world noise does in: no variant. Full tables (intake, survival, prediction error): `figures/exp6/tables6.md`. Figures: [figures/exp6](figures/exp6). Generated by `mvp/finalize6.py` from `report6.json`.

## Experiment 7: dark rooms under pe_cost (RQ3 re-run on Exp 5 forks)

ARCHITECTURE.md §15 after "make PE matter": rerun RQ3 — do self-built dark rooms appear under `pe_cost` while depletion is still on (the Friston question). Exp 5 built `pe_cost` but only ran Price + hidden-state recovery; Exp 6 changed the recovery yardstick and still got nulls. More seeds on Exp 6 would only tighten intervals around a consistent null, so tonight classifies every alive agent on the existing Exp 5 forks (step 2500).

- Same 7 variants × 3 seeds. Regime logic is identical to `metrics2.regimes` (v2 thresholds): resident = residence ≥ 100 and speed < 0.2 v_term; quiet via analytic OU variance; dark natural / other-built / self-built by wall counterfactuals. `env_var = 0.35`, `build_eff = 1.0`, layout index at t = 2500 with period 300 is 0.
- Paired contrasts by seed for the main arm `sw_pecost_pe` vs `sw_full`, `sw_pecost_pe_shuffled`, `pecost_pe_static`, `sw_no_cost`.
- Dark-rooming is a separate research question. **No agency claim is made here.**

**Self-built dark rooms do not appear under pe_cost while depletion is still on** (mean self-built fraction is 0 across pe_cost+depletion arms, or no seed shows any). Dark-rooming remains gated the way v2 found it: it shows up when starvation pressure is removed (see sw_no_cost), not when prediction error is charged. This is a dark-rooming result, **not** an agency claim.

  | variant | explore | settled-noisy | dark: natural | dark: other-built | dark: self-built | dark_any |
  |---|---|---|---|---|---|---|
  | sw_full | +0.9955 [+0.9843, +1.0066] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] |
  | sw_pe | +0.9937 [+0.9794, +1.0079] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0063 [-0.0079, +0.0206] | +0.0000 [+0.0000, +0.0000] | +0.0063 [-0.0079, +0.0206] |
  | sw_pecost | +0.9971 [+0.9846, +1.0096] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0029 [-0.0096, +0.0154] | +0.0000 [+0.0000, +0.0000] | +0.0029 [-0.0096, +0.0154] |
  | sw_pecost_pe | +0.9983 [+0.9911, +1.0055] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0017 [-0.0055, +0.0089] | +0.0000 [+0.0000, +0.0000] | +0.0017 [-0.0055, +0.0089] |
  | pecost_pe_static | +0.9899 [+0.9663, +1.0136] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0101 [-0.0136, +0.0337] | +0.0000 [+0.0000, +0.0000] | +0.0101 [-0.0136, +0.0337] |
  | sw_pecost_pe_shuffled | +0.9926 [+0.9607, +1.0245] | +0.0000 [+0.0000, +0.0000] | +0.0008 [-0.0027, +0.0044] | +0.0066 [-0.0217, +0.0349] | +0.0000 [+0.0000, +0.0000] | +0.0074 [-0.0245, +0.0393] |
  | sw_no_cost | +0.7118 [+0.6870, +0.7365] | +0.0000 [+0.0000, +0.0000] | +0.0682 [+0.0473, +0.0891] | +0.2201 [+0.2047, +0.2354] | +0.0000 [+0.0000, +0.0000] | +0.2882 [+0.2635, +0.3130] |

  | contrast | Δ dark_any | Δ self-built | Δ other-built | Δ natural | Δ mean surp | Δ wall mass |
  |---|---|---|---|---|---|---|
  | sw_pecost_pe − sw_full | -0.0029 [-0.0161, +0.0104] | +0.0000 [+0.0000, +0.0000] | -0.0029 [-0.0161, +0.0104] | +0.0000 [+0.0000, +0.0000] | -0.8771 [-2.5670, +0.8128] | -417.5172 [-2191.7577, +1356.7234] |
  | sw_pecost_pe − sw_pecost_pe_shuffled | -0.0057 [-0.0304, +0.0189] | +0.0000 [+0.0000, +0.0000] | -0.0049 [-0.0260, +0.0162] | -0.0008 [-0.0044, +0.0027] | -0.3683 [-1.2880, +0.5515] | +48.7998 [-1175.1789, +1272.7784] |
  | sw_pecost_pe − pecost_pe_static | -0.0084 [-0.0248, +0.0081] | +0.0000 [+0.0000, +0.0000] | -0.0084 [-0.0248, +0.0081] | +0.0000 [+0.0000, +0.0000] | -0.6877 [-2.8646, +1.4892] | -130.4388 [-386.5440, +125.6665] |
  | sw_pecost_pe − sw_no_cost | -0.2866 [-0.3185, -0.2546] | +0.0000 [+0.0000, +0.0000] | -0.2184 [-0.2392, -0.1975] | -0.0682 [-0.0891, -0.0473] | +5.6170 [+4.2972, +6.9367] | -8235.1567 [-10067.4913, -6402.8221] |

Main arm `sw_pecost_pe`: self-built +0.0000 [+0.0000, +0.0000], dark_any +0.0017 [-0.0055, +0.0089], mean surp +2.8587 [+1.7643, +3.9530], wall mass +2016.6434 [+102.4424, +3930.8445]. Full tables: `figures/exp7/tables7.md`. Figures: [figures/exp7](figures/exp7). Generated by `mvp/finalize7.py` from `report7.json`.

## Experiment 8: threshold sweep + dynamic CF_own (Exp 7 follow-up)

ARCHITECTURE.md §15 #3 and NIGHTLY_LOG 2026-11-05: after Exp 7's self-built = 0 under pe_cost+depletion, do a residence/speed threshold sweep and a true counterfactual re-simulation. Reuses Exp 5 forks (step 2500), 7 variants × 3 seeds.

- **(A) Threshold sweep:** res_min ∈ {25,50,100,200} × spd_frac ∈ {0.1,0.2,0.4,0.5} (default v2 = 100 / 0.2). Regime logic is parameterized `metrics2.regimes`.
- **(B) Dynamic CF:** for ≤40 dark / high-residence targets per run, replay T=100 closed-loop steps: **ref** (unchanged), **cf_own** (subtract scalar `own` from Wl at the agent's cell, clamp ≥0), **cf_cell** (zero the cell). Score leave of resident+quiet. **Honesty:** forks store scalar `own` (decayed local deposit EMA), not a lifetime deposit map — this is a dynamic CF with the own proxy, not a recovered lifetime field.
- Dark-rooming is a separate RQ. **No agency claim is made here.**

**(i) Threshold fragility:** Under pe_cost+depletion, the loosest thresholds invent only tiny self-built fractions (max mean across pe_cost arms ≈ 0.0042, a handful of agents). Default (100/0.2) stays at 0. Exp 7's self-built null is not qualitatively threshold-fragile — relaxing invents noise-scale counts, not a regime flip.

**(ii) CF_own reclass:** Dynamic CF_own (subtract scalar `own` from Wl at cell; not a lifetime deposit field) Δleave (own−ref) main arm sw_pecost_pe: -0.0083 [-0.0442, +0.0275]; sw_no_cost: +0.0417 [-0.0300, +0.1134]. Frac of fork-other-built reclassified behaviourally (CF_own leaves, ref stays): sw_pecost_pe +0.0000 [+0.0000, +0.0000], sw_no_cost +0.0519 [+0.0141, +0.0898].

  | variant | self-built @100/0.2 | self-built @25/0.5 | dark_any @100/0.2 | dark_any @25/0.5 | CF Δleave (own−ref) | frac reclass among other |
  |---|---|---|---|---|---|---|
  | sw_full | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] | +0.1807 [-0.0603, +0.4217] | +0.0167 [-0.0192, +0.0525] | +0.0000 [+0.0000, +0.0000] |
  | sw_pe | +0.0000 [+0.0000, +0.0000] | +0.0040 [-0.0054, +0.0134] | +0.0063 [-0.0079, +0.0206] | +0.1708 [-0.0492, +0.3908] | +0.0167 [-0.0550, +0.0884] | +0.0000 [+0.0000, +0.0000] |
  | sw_pecost | +0.0000 [+0.0000, +0.0000] | +0.0007 [-0.0024, +0.0039] | +0.0029 [-0.0096, +0.0154] | +0.1398 [+0.0164, +0.2633] | -0.0083 [-0.0442, +0.0275] | +0.0000 [+0.0000, +0.0000] |
  | sw_pecost_pe | +0.0000 [+0.0000, +0.0000] | +0.0042 [-0.0080, +0.0164] | +0.0017 [-0.0055, +0.0089] | +0.1465 [-0.0513, +0.3442] | -0.0083 [-0.0442, +0.0275] | +0.0000 [+0.0000, +0.0000] |
  | pecost_pe_static | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0101 [-0.0136, +0.0337] | +0.2088 [+0.0008, +0.4167] | -0.0250 [-0.0871, +0.0371] | +0.0000 [+0.0000, +0.0000] |
  | sw_pecost_pe_shuffled | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0074 [-0.0245, +0.0393] | +0.1671 [-0.0126, +0.3468] | +0.0083 [-0.0275, +0.0442] | +0.0417 [-0.1376, +0.2209] |
  | sw_no_cost | +0.0000 [+0.0000, +0.0000] | +0.0003 [-0.0004, +0.0010] | +0.2882 [+0.2635, +0.3130] | +0.7388 [+0.7149, +0.7627] | +0.0417 [-0.0300, +0.1134] | +0.0519 [+0.0141, +0.0898] |

Main arm `sw_pecost_pe`: self-built default +0.0000 [+0.0000, +0.0000], loose +0.0042 [-0.0080, +0.0164]; CF Δleave -0.0083 [-0.0442, +0.0275], reclass_other +0.0000 [+0.0000, +0.0000]. sw_no_cost CF Δleave +0.0417 [-0.0300, +0.1134], reclass_other +0.0519 [+0.0141, +0.0898]. Full tables: `figures/exp8/tables8.md`. Figures: [figures/exp8](figures/exp8). Generated by `mvp/finalize8.py` from `report8.json`.

## Experiment 9: exact deposit attribution (is Exp 7/8's self-built zero a proxy artifact?)

Exp 7/8 split dark rooms into self-built vs other-built with a scalar `own` proxy (decayed local EMA of the agent's deposits, discounted by movement), because the Exp 5 forks had no lifetime deposit maps. `mvp/sim7.py` adds an observer-only instrument: `Adep` (each live agent's lifetime wall deposits on the G×G grid, decayed exactly like the wall field) and `Aorph` (deposits of dead agents). **Checks:** sim7 with attribution off and on reproduces every sim5 state field bit for bit (main arm, 3 seeds, 300 steps, same harness). All 21 reruns to step 2500 equal the Exp 5 forks exactly in every dynamical field. ΣAdep + Aorph = Wl to ≤2e-06 relative (float32 summation order).

- **(i) Exact static classification:** `metrics2.regimes` with own = the agent's true surviving lifetime deposit at its cell, side by side with the proxy, at 100/0.2, at the sweep corners and on the full Exp 8 grid.
- **(ii) True dynamic CF:** T=100 closed-loop replay from step 2500 for ≤40 dark / high-residence targets per run. **ref** = unchanged; **cf_true** = delete the agent's whole lifetime deposit map from Wl; **cf_ctrl** = delete the same mass from walls ≥4 cells away, proportional to Wl. The key contrast is cf_true − cf_ctrl on leave (died/replaced, or no longer resident+quiet).
- Dark-rooming is a separate RQ. **No agency claim is made here.**

**Verdict:** Exact attribution, default 100/0.2: self-built CI excludes 0 in no variant; sw_no_cost exact self-built +0.0000 [+0.0000, +0.0000] vs proxy +0.0000 [+0.0000, +0.0000], dark_any +0.2882 [+0.2635, +0.3130]; max exact self-built over corners in pe_cost arms 0.0011. True CF (delete own lifetime deposits) − equal-mass control on leave: CI > 0 in no variant; sw_no_cost +0.0333 [-0.0959, +0.1626].

  | variant | self-built exact @100/0.2 | self-built proxy @100/0.2 | self-built exact @25/0.5 | dark_any @100/0.2 | Δleave true−ref | Δleave ctrl−ref | Δleave true−ctrl |
  |---|---|---|---|---|---|---|---|
  | sw_full | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] | +0.0083 [-0.0275, +0.0442] | +0.0000 [+0.0000, +0.0000] | +0.0083 [-0.0275, +0.0442] |
  | sw_pe | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0008 [-0.0028, +0.0044] | +0.0063 [-0.0079, +0.0206] | +0.0333 [-0.0025, +0.0692] | +0.0083 [-0.0275, +0.0442] | +0.0250 [-0.0371, +0.0871] |
  | sw_pecost | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0029 [-0.0096, +0.0154] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
  | sw_pecost_pe | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0011 [-0.0035, +0.0057] | +0.0017 [-0.0055, +0.0089] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
  | pecost_pe_static | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0101 [-0.0136, +0.0337] | -0.0083 [-0.0442, +0.0275] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.0442, +0.0275] |
  | sw_pecost_pe_shuffled | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0074 [-0.0245, +0.0393] | +0.0083 [-0.0275, +0.0442] | +0.0083 [-0.0275, +0.0442] | +0.0000 [+0.0000, +0.0000] |
  | sw_no_cost | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0005, +0.0009] | +0.2882 [+0.2635, +0.3130] | +0.0417 [-0.0532, +0.1365] | +0.0083 [-0.0634, +0.0800] | +0.0333 [-0.0959, +0.1626] |

Full tables (all corners, own-share, relaxed-threshold CF, paired contrasts): `figures/exp9/tables9.md`. Figures: [figures/exp9](figures/exp9). Generated by `mvp/finalize9.py` from `report9.json`.

## Experiment 10: wall-protected intake (does making walls privately valuable produce self-built dark rooms?)

Exp 9 showed exact self-built dark rooms are zero: walls act as a public good. `mvp/sim8.py` = sim7 + traced param `guard`, which feeds the exact attribution map back into the dynamics. For agent i in cell c, claim_i = guard · clip(Adep[i,c]/Wl[c], 0, 1) · Wl[c]/(Wl[c]+0.5). The claimed fraction of i's eat demand is served first and is immune to F, the rest competes for what is left, and theft from i is scaled by (1 − claim_i). Claims follow wall mass, not F-damping, so `sw_guard_inert` (build_eff 0) gives private claims with no shelter. **Checks:** with guard=0, sim8 reproduces all 54 sim7 state fields bit for bit (sw_full and sw_pecost_pe, 3 seeds, 300 steps), and the guard=0 arms' step-2500 forks equal the Exp 9 forks exactly. The exact Price identity residual is ≤3e-16.

Arms (switching world, Exp 5 base, seeds 0–2, 3000 steps): sw_full, sw_guard, sw_guard_inert, sw_guard_shuffled (heredity broken), sw_pecost_pe, sw_pecost_pe_guard. Price trait = lifetime mean build action; ΣS = selection summed over boundaries t ≥ 1000. The true CF (T=100 from step 2500) deletes the agent's lifetime deposits *and its ownership of them*; the control deletes equal mass ≥4 cells away. Dark-rooming is a separate RQ. **No agency claim is made here.**

**Verdict:** Mean build action higher (CI>0) in: none; lower in: none. Price selection on build (Σ S over t≥1000) CI>0 in: none; CI<0 in: none. Exact self-built @100/0.2 CI>0 in: none (max over corners in guard arms 0.0015). True CF beats equal-mass control on leave in: none; losing own walls cuts intake (CI<0) in: none. Price identity residual ≤ 3.2e-16.

  | arm | build (t≥1000) | claim | own share here | Price ΣS build | self-built exact @100/0.2 | dark_any | CF Δleave true−ctrl | CF Δintake true−ctrl |
  |---|---|---|---|---|---|---|---|---|
  | sw_full | +0.5162 [+0.1781, +0.8543] | +0.0000 [+0.0000, +0.0000] | +0.0809 [+0.0281, +0.1337] | +0.0472 [-0.2262, +0.3206] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] | +0.0083 [-0.0275, +0.0442] | +0.0000 [-0.0022, +0.0022] |
  | sw_guard | +0.4714 [-0.0724, +1.0152] | +0.0434 [-0.0107, +0.0974] | +0.0753 [+0.0073, +0.1433] | +0.0016 [-0.1416, +0.1448] | +0.0000 [+0.0000, +0.0000] | +0.0048 [-0.0106, +0.0202] | +0.0083 [-0.0275, +0.0442] | -0.0002 [-0.0039, +0.0034] |
  | sw_guard_inert | +0.4682 [+0.2466, +0.6898] | +0.0119 [+0.0042, +0.0196] | +0.0216 [+0.0111, +0.0321] | -0.0436 [-0.3182, +0.2311] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0004 [-0.0021, +0.0029] |
  | sw_guard_shuffled | +0.4637 [+0.3753, +0.5522] | +0.0562 [+0.0348, +0.0777] | +0.0995 [+0.0662, +0.1328] | -0.0493 [-0.1534, +0.0549] | +0.0000 [+0.0000, +0.0000] | +0.0177 [-0.0196, +0.0551] | +0.0500 [-0.1143, +0.2143] | +0.0006 [-0.0007, +0.0019] |
  | sw_pecost_pe | +0.4916 [+0.1448, +0.8384] | +0.0000 [+0.0000, +0.0000] | +0.0895 [+0.0555, +0.1235] | -0.0609 [-0.2607, +0.1389] | +0.0000 [+0.0000, +0.0000] | +0.0017 [-0.0055, +0.0089] | +0.0000 [+0.0000, +0.0000] | +0.0003 [-0.0061, +0.0067] |
  | sw_pecost_pe_guard | +0.4679 [+0.0542, +0.8815] | +0.0454 [+0.0019, +0.0890] | +0.0878 [+0.0383, +0.1373] | -0.0358 [-0.2183, +0.1467] | +0.0000 [+0.0000, +0.0000] | +0.0035 [-0.0059, +0.0129] | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0028, +0.0028] |

  | paired contrast | build | own share here | Price ΣS build | self-built exact | CF true−ctrl |
  |---|---|---|---|---|---|
  | sw_guard - sw_full | -0.0448 [-0.2507, +0.1611] | -0.0056 [-0.0227, +0.0114] | -0.0456 [-0.2080, +0.1168] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.0621, +0.0621] |
  | sw_guard_inert - sw_full | -0.0480 [-0.2129, +0.1169] | -0.0593 [-0.1023, -0.0163] | -0.0908 [-0.1398, -0.0418] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.0442, +0.0275] |
  | sw_guard - sw_guard_shuffled | +0.0077 [-0.4483, +0.4637] | -0.0242 [-0.1031, +0.0547] | +0.0509 [-0.0313, +0.1330] | +0.0000 [+0.0000, +0.0000] | -0.0417 [-0.2209, +0.1376] |
  | sw_guard - sw_guard_inert | +0.0032 [-0.3482, +0.3546] | +0.0537 [-0.0050, +0.1125] | +0.0452 [-0.1004, +0.1907] | +0.0000 [+0.0000, +0.0000] | +0.0083 [-0.0275, +0.0442] |
  | sw_pecost_pe_guard - sw_pecost_pe | -0.0237 [-0.1211, +0.0737] | -0.0017 [-0.0299, +0.0264] | +0.0251 [-0.0394, +0.0895] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Full tables: `figures/exp10/tables10.md`. Figures: [figures/exp10](figures/exp10). Generated by `mvp/finalize10.py` from `report10.json`.

## Experiment 11: guard dose-response with mass-based ownership (does building pay once the private return can beat its cost?)

Exp 10's share-based claim stayed tiny (≈0.05) because walls are a dense commons, so the private return never clearly beat the build cost. `mvp/sim9.py` = sim8 + `claim_mode="mass"`: claim_i = min(guard · own_i/(own_i + 0.05), 0.95), where own_i is i's exact lifetime deposit in its cell (0.05 ≈ the median measured on Exp 10 sw_full forks at step 2500). It also logs an exact one-step private return `premstep`: i's energy income under the guard rules minus its income under guard=0 rules from the same state (harvest, theft avoided, change in own theft take). This is observer-only, computed behind an optimization barrier. **Checks:** with guard=0, sim9 reproduces all sim8 state fields bit for bit (3 seeds, 300 steps; an earlier version without the barrier diverged by 1 ulp at step 2 through XLA re-fusion, so it was fixed before any run), the sw_full step-2500 forks equal the Exp 10 forks exactly, and the Price identity residual is ≤4e-16.

Arms (switching world, Exp 5 base, seeds 0–2, 3000 steps): sw_full (guard 0), sw_mg05, sw_mg1, sw_mg2 (guard 0.5/1/2), sw_mg2_shuffled (heredity broken), sw_mg2_inert (guard 2, build_eff 0: private claims, no shelter). The marginal return is the per-seed OLS slope of lifetime mean private return on lifetime mean build (agents ≥100 steps old at step 2500), compared with c_build = 0.003. Exact self-built classification and the true CF follow the Exp 10 protocol. Dark-rooming is a separate RQ. **No agency claim is made here.**

**Verdict:** Mean build action higher (CI>0) in: none; lower in: none. Price selection on build (Σ S over t≥1000) CI>0 in: none; CI<0 in: none. Exact self-built @100/0.2 CI>0 in: none (max over corners in guard arms 0.0007). True CF beats equal-mass control on leave in: none; losing own walls cuts intake (CI<0) in: none. Mean private return − build cost CI>0 in: sw_mg1, sw_mg2, sw_mg2_shuffled, sw_mg2_inert; marginal return of building (slope of lifetime prem on lifetime build) CI-low > c_build=0.003 in: sw_mg2_shuffled. Dose slope of build per unit guard +0.0293 [-0.0876, +0.1462]; of ΣS(build) +0.0412 [-0.0693, +0.1516]. Price identity residual ≤ 3.6e-16.

  | arm | build (t≥1000) | claim | private return | build cost | Price ΣS build | self-built exact @100/0.2 | dark_any | CF Δleave true−ctrl | CF Δintake true−ctrl |
  |---|---|---|---|---|---|---|---|---|---|
  | sw_full | +0.5162 [+0.1781, +0.8543] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0015 [+0.0005, +0.0026] | +0.0472 [-0.2262, +0.3206] | +0.0000 [+0.0000, +0.0000] | +0.0045 [-0.0066, +0.0157] | +0.0083 [-0.0275, +0.0442] | +0.0000 [-0.0022, +0.0022] |
  | sw_mg05 | +0.5928 [+0.4874, +0.6982] | +0.2237 [+0.1829, +0.2645] | +0.0017 [+0.0011, +0.0022] | +0.0018 [+0.0015, +0.0021] | +0.0517 [-0.3909, +0.4944] | +0.0000 [+0.0000, +0.0000] | +0.0237 [-0.0417, +0.0891] | -0.0417 [-0.0775, -0.0058] | -0.0009 [-0.0052, +0.0034] |
  | sw_mg1 | +0.5774 [+0.3893, +0.7655] | +0.4610 [+0.2807, +0.6412] | +0.0044 [+0.0043, +0.0044] | +0.0017 [+0.0012, +0.0023] | +0.1954 [-0.3069, +0.6976] | +0.0000 [+0.0000, +0.0000] | +0.0290 [-0.0259, +0.0839] | +0.0083 [-0.0275, +0.0442] | +0.0002 [-0.0069, +0.0073] |
  | sw_mg2 | +0.5919 [+0.5469, +0.6370] | +0.7088 [+0.6151, +0.8026] | +0.0072 [+0.0050, +0.0094] | +0.0018 [+0.0016, +0.0019] | +0.1123 [-0.0329, +0.2575] | +0.0000 [+0.0000, +0.0000] | +0.0142 [+0.0017, +0.0267] | +0.0000 [-0.1076, +0.1076] | -0.0006 [-0.0097, +0.0085] |
  | sw_mg2_shuffled | +0.5129 [+0.4654, +0.5604] | +0.6892 [+0.6595, +0.7190] | +0.0065 [+0.0049, +0.0081] | +0.0015 [+0.0014, +0.0017] | +0.0097 [-0.0707, +0.0900] | +0.0000 [+0.0000, +0.0000] | +0.0386 [-0.0012, +0.0784] | -0.0250 [-0.0871, +0.0371] | -0.0006 [-0.0033, +0.0020] |
  | sw_mg2_inert | +0.5655 [+0.3048, +0.8261] | +0.3886 [+0.2275, +0.5497] | +0.0055 [+0.0048, +0.0062] | +0.0017 [+0.0009, +0.0025] | +0.0474 [-0.1443, +0.2390] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0003 [-0.0005, +0.0012] |

  | paired contrast | build | net return | Price ΣS build | self-built exact | CF true−ctrl |
  |---|---|---|---|---|---|
  | sw_mg05 - sw_full | +0.0766 [-0.2286, +0.3818] | +0.0014 [+0.0008, +0.0020] | +0.0045 [-0.1862, +0.1953] | +0.0000 [+0.0000, +0.0000] | -0.0500 [-0.0500, -0.0500] |
  | sw_mg1 - sw_full | +0.0612 [-0.1495, +0.2719] | +0.0042 [+0.0035, +0.0048] | +0.1482 [-0.1626, +0.4589] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
  | sw_mg2 - sw_full | +0.0757 [-0.2293, +0.3808] | +0.0070 [+0.0054, +0.0086] | +0.0651 [-0.1409, +0.2710] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.1518, +0.1351] |
  | sw_mg2 - sw_mg2_shuffled | +0.0791 [-0.0063, +0.1644] | +0.0005 [-0.0016, +0.0026] | +0.1027 [+0.0270, +0.1783] | +0.0000 [+0.0000, +0.0000] | +0.0250 [-0.0371, +0.0871] |
  | sw_mg2 - sw_mg2_inert | +0.0265 [-0.1904, +0.2433] | +0.0016 [-0.0006, +0.0039] | +0.0649 [-0.0929, +0.2228] | +0.0000 [+0.0000, +0.0000] | +0.0000 [-0.1076, +0.1076] |
  | sw_mg2_inert - sw_full | +0.0493 [-0.1127, +0.2113] | +0.0054 [+0.0044, +0.0063] | +0.0001 [-0.0851, +0.0854] | +0.0000 [+0.0000, +0.0000] | -0.0083 [-0.0442, +0.0275] |

Full tables: `figures/exp11/tables11.md`. Figures: [figures/exp11](figures/exp11). Generated by `mvp/finalize11.py` from `report11.json`.

## Experiment 12: pre-registered seed extension of Exp 11 (is the heredity contrast real?)

Exp 11 left one lead at n = 3: with mass-based private walls at guard 2, Price selection on building was stronger with heredity intact than broken (sw_mg2 − sw_mg2_shuffled, ΣS build +0.103 [+0.027, +0.178]). Exp 12 adds seeds 3–8 for sw_full, sw_mg2 and sw_mg2_shuffled with the identical sim9 harness (3000 steps, two vmapped batches of 3 seeds). The contrasts were fixed before running: **primary** = sw_mg2 − sw_mg2_shuffled on Price ΣS(build) over t ≥ 1000 and on mean build over t ≥ 1000, paired by seed, n = 9.

  | contrast (paired, n = 9) | Price ΣS build | mean build | ΣΔz̄ build |
  |---|---|---|---|
  | sw_mg2 − sw_mg2_shuffled (primary) | **+0.146 [+0.027, +0.264]**, p = 0.022 | +0.047 [−0.003, +0.097], p = 0.063 | +0.039 [+0.005, +0.072] |
  | same, new seeds 3–8 only (n = 6) | +0.167 [−0.032, +0.366] | +0.031 [−0.047, +0.109] | +0.029 [−0.025, +0.082] |
  | sw_mg2 − sw_full | +0.156 [−0.025, +0.336] | +0.068 [+0.012, +0.125] | −0.011 [−0.083, +0.061] |
  | sw_mg2_shuffled − sw_full | +0.010 [−0.079, +0.099] | +0.021 [−0.064, +0.106] | −0.050 [−0.107, +0.008] |

**Reading.** The selection half of the lead holds: with private walls and heredity intact, building is positively selected relative to the broken-heredity control (8 of 9 seeds positive). The trait-response half doesn't clear its CI: mean build is not reliably higher than in the shuffled control, because within sw_mg2 the selection term is largely offset by a negative within-individual term (ΣW build −0.141 [−0.287, +0.005]; net ΣΔz̄ +0.016 [−0.043, +0.076]). The new seeds alone don't reach significance (n = 6), so this is a modest, not a robust, effect. Price residual ≤ 2.7e-16. This is selection on a niche-construction trait, not agency, and no agency claim is made. Tables: `figures/exp12/tables12.md`; figure: `figures/exp12/exp12_contrasts.png`.

## Experiment 13: long horizon (does selection on building become more building?)

Exp 12 found that with mass-based private walls (guard 2), building is selected more when heredity is intact than when it's broken. But mean build barely moved in 3000 steps, because a negative within-individual term cancelled the selection. Exp 13 runs the same sim9 harness to 10000 steps, seeds 0–8, arms sw_full, sw_mg2 and sw_mg2_shuffled, with on-disk checkpoints every 1000 steps (resume is bit-for-bit, `figures/exp13/resumecheck.json`). Seeds 0–2 reproduce the Exp 11 series exactly up to t = 3000. **Pre-registered test (fixed 2026-11-12):** sw_mg2 − sw_mg2_shuffled on mean build over t ≥ 5000, paired by seed, n = 9, 95% t CI > 0.

  | contrast (paired, n = 9, t ≥ 5000) | mean build | Price ΣS build | ΣΔz̄ build | wall mass |
  |---|---|---|---|---|
  | sw_mg2 − sw_mg2_shuffled (pre-registered) | **+0.011 [−0.045, +0.068]**, p = 0.65 | +0.206 [−0.144, +0.556] | −0.040 [−0.138, +0.058] | +216 [−28, +460] |
  | sw_mg2 − sw_full | +0.028 [−0.086, +0.142] | +0.364 [−0.053, +0.782] | −0.005 [−0.090, +0.080] | +953 [+374, +1531] |
  | sw_mg2_shuffled − sw_full | +0.016 [−0.122, +0.155] | +0.158 [−0.042, +0.359] | +0.035 [−0.103, +0.173] | +736 [+90, +1383] |

  Not pre-registered, window by window (sw_mg2 − sw_mg2_shuffled, mean build): 0–2k +0.025 [−0.008, +0.059]; 2–4k +0.062 [+0.010, +0.114]; 4–6k +0.042 [+0.007, +0.077]; 6–8k +0.014 [−0.037, +0.065]; 8–10k −0.001 [−0.084, +0.082].

**Reading.** The pre-registered test fails. With heredity intact, building does pull ahead of the shuffled control for a while (2000–6000 steps, a post-hoc window, 15 window tests in total, so treat it as a hint). Then the gap closes, and over the late half mean build is the same in all three arms (≈0.49–0.52). The selection term stays positive on average but becomes noisy (seed 5 +1.07, seed 8 −0.64) and isn't converted into a lasting trait difference. The guard raises total wall mass and population in both guard arms whether or not heredity is intact, so private walls change the ecology rather than producing a heritable building strategy. Taken with Exp 10–12: in this world, making walls privately valuable produces a transient, modest selection effect on building and no durable niche-construction strategy. Price residual ≤ 3.8e-16. No agency claim. Tables: `figures/exp13/tables13.md`; figure: `figures/exp13/exp13_trajectory.png`.

## Reproduce
```bash
./setup.sh   # or setup.ps1 on Windows
cd mvp
../.venv/bin/python exp2.py main --out runs2/main --seeds 3 --steps 4000 --batch 3 --procs 6 && ../.venv/bin/python analyze2.py
../.venv/bin/python exp3_price.py bank && ../.venv/bin/python exp3_price.py replay --procs 6 && ../.venv/bin/python exp3_perturb.py --procs 6 && ../.venv/bin/python analyze3.py
../.venv/bin/python exp4_price.py bank && ../.venv/bin/python exp4_price.py run --procs 7 && ../.venv/bin/python exp4_perturb.py --procs 8 && ../.venv/bin/python analyze4.py
../.venv/bin/python exp5_price.py bank && ../.venv/bin/python exp5_price.py run --procs 5 && ../.venv/bin/python exp5_perturb.py --procs 5 && ../.venv/bin/python analyze5.py
../.venv/bin/python exp6_regulate.py --procs 6 && ../.venv/bin/python analyze6.py   # needs the Exp 5 forks in runs5/
../.venv/bin/python exp7_darkroom.py --procs 6 && ../.venv/bin/python analyze7.py   # needs the Exp 5 forks in runs5/
../.venv/bin/python exp8_counterfactual.py --procs 4 && ../.venv/bin/python analyze8.py   # needs the Exp 5 forks in runs5/
../.venv/bin/python exp13_run.py --resumecheck && for B in 0,1,2 3,4,5 6,7,8; do EXP13_SEEDS=$B ../.venv/bin/python exp13_run.py --procs 3; done && ../.venv/bin/python analyze13.py   # needs runs11/ for the Exp 11 reproducibility check
../.venv/bin/python exp9_attr.py --bitcheck && ../.venv/bin/python exp9_attr.py --procs 4 && ../.venv/bin/python exp9_cf.py --procs 4 && ../.venv/bin/python analyze9.py   # exp9_attr checks against the Exp 5 forks in runs5/
```
Long runs checkpoint to disk, one file per batch, so they can be resumed. Run dumps (`runs*/`) and videos aren't in the repo because of their size.

## Caveats
- λ is conditional on inputs.
- n = 3 seeds and small, short CPU runs.
- The individuality and empowerment measures are simplified Gaussian proxies.
- `modal_sweep.py` (GPU) is untested.
- See ARCHITECTURE.md §14.
