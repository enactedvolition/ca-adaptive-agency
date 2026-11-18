# Experiment 16 tables

Bit-for-bit (rep_free=0 vs sim11 all fields, sim9 all fields, seeds 0-4 x 300 steps): **True**

## Pay-check, start = fork (pre-registered primary)

| thr | pop aware | pop blind | Δpop (per seed) | Δnet intake (per seed) | births/step aware / blind | mean E aware / blind |
|---|---|---|---|---|---|---|
| 0.8 | 534 | 500 | +6.9% (+1.9%, +13.0%, +5.8%) | -13.0% (-15.3%, -14.4%, -9.2%) | 3.48 / 2.03 | 0.35 / 0.59 |
| 1.2 | 500 | 454 | +10.8% (+4.7%, +25.5%, +2.1%) | -13.0% (-13.1%, -12.9%, -13.1%) | 1.67 / 1.21 | 0.48 / 0.85 |
| 1.6 | 510 | 403 | +26.9% (+21.9%, +40.1%, +18.6%) | -13.3% (-12.0%, -13.4%, -14.5%) | 1.09 / 0.74 | 0.59 / 1.04 |

Unmodified brains, rep_free=1: pop 563, 644, 616, mean E 0.19, 0.18, 0.19, rep>0 0.81, 0.69, 0.57; rep_free=0: pop 541, 497, 482.
Best thr by population: 0.8. **PASS (pre-registered): False**; PASS (literal suggested): False

## Pay-check, start = init (secondary)

| thr | pop aware | pop blind | Δpop (per seed) | Δnet intake (per seed) | births/step aware / blind | mean E aware / blind |
|---|---|---|---|---|---|---|
| 0.8 | 591 | 468 | +26.4% (+21.8%, +27.2%, +30.3%) | -7.0% (-8.1%, -7.2%, -5.7%) | 4.74 / 2.11 | 0.38 / 0.52 |
| 1.2 | 468 | 438 | +6.9% (+8.2%, +10.1%, +2.4%) | +3.0% (-0.3%, +4.6%, +4.9%) | 2.18 / 1.40 | 0.52 / 0.64 |
| 1.6 | 437 | 375 | +16.6% (+23.2%, +19.7%, +6.9%) | -1.5% (-2.2%, -1.4%, -1.0%) | 1.10 / 0.73 | 0.61 / 0.88 |

Unmodified brains, rep_free=1: pop 678, 552, 788, mean E 0.18, 0.23, 0.19, rep>0 0.64, 0.42, 0.58; rep_free=0: pop 441, 411, 424.
Best thr by population: 0.8. **PASS (pre-registered): False**; PASS (literal suggested): False

## Mixed competition (post hoc / exploratory): aware lineages (even root) vs always-reproduce (odd root)

| thr | share t=0 | share last 50 | Δshare | births aware / always |
|---|---|---|---|---|
| 0.8 (s0) | 0.893 | 0.853 | -0.040 | 1557 / 755 |
| 0.8 (s1) | 0.695 | 0.295 | -0.399 | 703 / 3136 |
| 0.8 (s2) | 0.724 | 0.466 | -0.257 | 900 / 2203 |
| 1.2 (s0) | 0.893 | 0.698 | -0.195 | 621 / 1664 |
| 1.2 (s1) | 0.695 | 0.140 | -0.554 | 238 / 3962 |
| 1.2 (s2) | 0.724 | 0.267 | -0.456 | 332 / 3194 |
| 1.6 (s0) | 0.893 | 0.456 | -0.437 | 256 / 2230 |
| 1.6 (s1) | 0.695 | 0.101 | -0.593 | 91 / 4171 |
| 1.6 (s2) | 0.724 | 0.122 | -0.601 | 105 / 4071 |
