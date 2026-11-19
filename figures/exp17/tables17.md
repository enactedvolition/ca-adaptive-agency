# Exp 17 tables

## Mixed-competition gate (aware = rep iff E > THR; C = E_endow + E_over = 0.792)

| THR | parity | vs | dshare s0 | s1 | s2 | mean [95% CI] | all > 0 |
|---|---|---|---|---|---|---|---|
| 0.8 | even=aware | always | +0.107 | +0.304 | +0.276 | +0.229 [-0.037, +0.495] | True |
| 0.8 | even=aware | blind | +0.077 | +0.189 | +0.247 | +0.171 [-0.043, +0.385] | True |
| 0.8 | odd=aware (post hoc) | always | +0.892 | +0.694 | +0.723 | +0.770 [+0.504, +1.036] | True |
| 0.8 | odd=aware (post hoc) | blind | +0.405 | +0.497 | +0.443 | +0.448 [+0.332, +0.564] | True |
| 1.0 | even=aware | always | +0.107 | +0.305 | +0.276 | +0.229 [-0.037, +0.496] | True |
| 1.0 | even=aware | blind | +0.077 | +0.182 | +0.205 | +0.155 [-0.015, +0.324] | True |
| 1.2 | even=aware | always | +0.107 | +0.304 | +0.276 | +0.229 [-0.037, +0.495] | True |
| 1.2 | even=aware | blind | +0.079 | +0.110 | +0.206 | +0.132 [-0.032, +0.295] | True |
| 1.6 | even=aware | always | +0.106 | +0.305 | +0.276 | +0.229 [-0.039, +0.497] | True |
| 1.6 | even=aware | blind | +0.041 | +0.106 | +0.159 | +0.102 [-0.045, +0.249] | True |

Pre-registered gate (THR 0.8, both contrasts, 3/3 seeds > 0): **PASS**

## Evolution (3000 steps; exact Price decomposition, D = S + T + W + I + resid)

| arm | n seeds | pop (last 500) | rep–E corr (last 500) | z̄ rep-sens c_esr/l0n at 3000 | ΣS rep-sens | ΣS net intake | max abs resid |
|---|---|---|---|---|---|---|---|
| sw_full | 9 | +530.5 [+497.4, +563.6] | -0.01201 [-0.05327, +0.02926] | +0.02664 [-0.05842, +0.1117] | +0.02087 [-0.07989, +0.1216] | +0.1295 [+0.1107, +0.1482] | 1.27e-16 |
| sw_pays | 9 | +460.4 [+390.7, +530.2] | -0.07762 [-0.1071, -0.04811] | +0.07189 [-0.01402, +0.1578] | +0.1696 [-0.06797, +0.4073] | +0.4892 [+0.4022, +0.5761] | 1.31e-16 |
| sw_pays_blind | 9 | +521.7 [+422.2, +621.1] | -0.07137 [-0.07857, -0.06418] | +0 [+0, +0] | +0 [+0, +0] | +0.5502 [+0.4606, +0.6399] | 3.34e-17 |

Seed-matched contrasts (same seed keys; mean [95% t CI], n = 9):

- sw_pays − sw_pays_blind, pop: -61.21 [-162.9, +40.52]
- sw_pays − sw_pays_blind, cor: -0.006246 [-0.03428, +0.02179]
- sw_pays − sw_pays_blind, za: +0.07189 [-0.01402, +0.1578]
- sw_pays − sw_pays_blind, sa: +0.1696 [-0.06797, +0.4073]
- sw_pays − sw_pays_blind, E: -0.08693 [-0.3222, +0.1484]
- sw_pays − sw_pays_blind, rep: +0.001207 [-0.005075, +0.007489]
- sw_pays − sw_full, pop: -70.07 [-147.9, +7.725]
- sw_pays − sw_full, cor: -0.06561 [-0.1022, -0.029]
- sw_pays − sw_full, za: +0.04525 [-0.07695, +0.1675]
- sw_pays − sw_full, sa: +0.1488 [-0.08739, +0.3849]
- sw_pays − sw_full, E: +0.04423 [-0.06113, +0.1496]
- sw_pays − sw_full, rep: -0.5821 [-0.6937, -0.4706]
- sw_full: mean E (last 500) 0.676, rep-action>0 fraction 0.5973
- sw_pays: mean E (last 500) 0.720, rep-action>0 fraction 0.0152
- sw_pays_blind: mean E (last 500) 0.807, rep-action>0 fraction 0.0140
