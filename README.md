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
| Exp 4 (`sim4.py`, `exp4_*.py`) | Make regulation pay: (a) agents sense their own prediction error, (b) instability costs energy | see below |

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

## Reproduce
```bash
./setup.sh   # or setup.ps1 on Windows
cd mvp
../.venv/bin/python exp2.py main --out runs2/main --seeds 3 --steps 4000 --batch 3 --procs 6 && ../.venv/bin/python analyze2.py
../.venv/bin/python exp3_price.py bank && ../.venv/bin/python exp3_price.py replay --procs 6 && ../.venv/bin/python exp3_perturb.py --procs 6 && ../.venv/bin/python analyze3.py
../.venv/bin/python exp4_price.py bank && ../.venv/bin/python exp4_price.py run --procs 7 && ../.venv/bin/python exp4_perturb.py --procs 8 && ../.venv/bin/python analyze4.py
```
Long runs checkpoint to disk, one file per batch, so they can be resumed. Run dumps (`runs*/`) and videos aren't in the repo because of their size.

## Caveats
- λ is conditional on inputs.
- n = 3 seeds and small, short CPU runs.
- The individuality and empowerment measures are simplified Gaussian proxies.
- `modal_sweep.py` (GPU) is untested.
- See ARCHITECTURE.md §14.
