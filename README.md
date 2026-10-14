# ca-adaptive-agency

A JAX agent world with free-moving particles. Each particle has a small random recurrent brain (an Elman RNN: 16 sense inputs (15 in v1–v3), 16 hidden units, 8 actions), an energy budget, mutation, and an environment it changes: resources deplete and regrow, agents deposit walls and pheromone, and the weather is noisy.

The question is whether anything here earns the words **edge of chaos**, **agency**, **dark room**, or **self-regulation** once controls are in place. Every claim is gated against controls: shuffled heredity, no mutation, no energy cost, static environment, and inert walls. Statistics are 95% t-intervals over 3 seeds (df = 2, so the intervals are wide). Nulls are reported as nulls. All numbers come from the simulations in this repo; there is no synthetic data.

Design and full results: [`ARCHITECTURE.md`](ARCHITECTURE.md). Licence: Apache-2.0.

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

RESULTS_PLACEHOLDER

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install "jax[cpu]" numpy scipy matplotlib zarr
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
