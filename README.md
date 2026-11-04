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

## Reproduce
```bash
./setup.sh   # or setup.ps1 on Windows
cd mvp
../.venv/bin/python exp2.py main --out runs2/main --seeds 3 --steps 4000 --batch 3 --procs 6 && ../.venv/bin/python analyze2.py
../.venv/bin/python exp3_price.py bank && ../.venv/bin/python exp3_price.py replay --procs 6 && ../.venv/bin/python exp3_perturb.py --procs 6 && ../.venv/bin/python analyze3.py
../.venv/bin/python exp4_price.py bank && ../.venv/bin/python exp4_price.py run --procs 7 && ../.venv/bin/python exp4_perturb.py --procs 8 && ../.venv/bin/python analyze4.py
../.venv/bin/python exp5_price.py bank && ../.venv/bin/python exp5_price.py run --procs 5 && ../.venv/bin/python exp5_perturb.py --procs 5 && ../.venv/bin/python analyze5.py
../.venv/bin/python exp6_regulate.py --procs 6 && ../.venv/bin/python analyze6.py   # needs the Exp 5 forks in runs5/
```
Long runs checkpoint to disk, one file per batch, so they can be resumed. Run dumps (`runs*/`) and videos aren't in the repo because of their size.

## Caveats
- λ is conditional on inputs.
- n = 3 seeds and small, short CPU runs.
- The individuality and empowerment measures are simplified Gaussian proxies.
- `modal_sweep.py` (GPU) is untested.
- See ARCHITECTURE.md §14.
