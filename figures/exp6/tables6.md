# Experiment 6: recovery scored on regulated variables — tables (per-run means over targets; 95% t-CI over seeds, n=3 → df=2)

## Energy-drop test (50% of energy removed at t=0; recovery fraction r = 1 − mean deficit over steps 60–120 / amount removed)

| variant | seeds | r closed | r clamped (passive) | r sensor-blind | A = r_cl − r_clamp | B = r_cl − r_blind | survival at 120 (cl / clamp / blind, targets whose reference survives; last = reference survival over all targets) |
|---|---|---|---|---|---|---|---|
| sw_full | 3 | +0.746 [+0.514, +0.979] | +0.667 [-0.066, +1.401] | +0.637 [+0.087, +1.187] | +0.079 [-0.831, +0.989] | +0.109 [-0.663, +0.881] | 0.87 / 0.88 / 0.84 / 0.71 |
| sw_pe | 3 | +0.558 [+0.120, +0.996] | +0.663 [+0.104, +1.223] | +0.819 [+0.179, +1.459] | -0.106 [-0.377, +0.165] | -0.261 [-0.764, +0.242] | 0.91 / 0.92 / 0.89 / 0.78 |
| sw_pecost | 3 | +0.455 [-0.446, +1.356] | +0.765 [-0.132, +1.662] | +0.520 [-0.371, +1.410] | -0.310 [-0.409, -0.211] | -0.065 [-0.540, +0.411] | 0.89 / 0.92 / 0.90 / 0.73 |
| sw_pecost_pe | 3 | +0.543 [+0.029, +1.057] | +0.819 [+0.433, +1.205] | +0.483 [-0.656, +1.622] | -0.276 [-0.430, -0.121] | +0.060 [-1.299, +1.419] | 0.91 / 0.92 / 0.95 / 0.73 |
| pecost_pe_static | 3 | +0.352 [-0.775, +1.480] | +0.657 [+0.270, +1.044] | +0.396 [-0.124, +0.916] | -0.304 [-1.222, +0.613] | -0.043 [-0.967, +0.881] | 0.81 / 0.86 / 0.77 / 0.61 |
| sw_pecost_pe_shuffled | 3 | +1.206 [-0.886, +3.298] | +0.899 [+0.080, +1.718] | +0.773 [-0.542, +2.087] | +0.307 [-1.132, +1.746] | +0.434 [-0.469, +1.336] | 0.79 / 0.87 / 0.89 / 0.70 |
| sw_no_cost | 3 | +1.093 [+0.817, +1.368] | +0.967 [+0.765, +1.169] | +0.977 [+0.844, +1.111] | +0.125 [+0.035, +0.216] | +0.115 [-0.108, +0.338] | 1.00 / 1.00 / 1.00 / 1.00 |

## Robustness: median of per-target A, and fraction of targets where the closed loop ends with the smaller deficit (0.5 = chance)

| variant | targets used per seed | median A | P(closed deficit < clamped deficit) |
|---|---|---|---|
| sw_full | 27,28,30 | +0.013 [-0.173, +0.199] | +0.484 [+0.327, +0.642] |
| sw_pe | 24,34,24 | +0.022 [-0.132, +0.176] | +0.462 [+0.357, +0.567] |
| sw_pecost | 20,29,29 | -0.123 [-0.443, +0.197] | +0.386 [+0.212, +0.560] |
| sw_pecost_pe | 21,29,19 | -0.014 [-0.061, +0.032] | +0.466 [+0.428, +0.504] |
| pecost_pe_static | 14,28,16 | -0.041 [-0.302, +0.220] | +0.452 [+0.247, +0.657] |
| sw_pecost_pe_shuffled | 24,30,29 | +0.024 [-0.121, +0.169] | +0.495 [+0.431, +0.559] |
| sw_no_cost | 40,40,40 | +0.020 [-0.091, +0.131] | +0.525 [+0.214, +0.836] |

## Extra cumulative intake by step 120 relative to the unperturbed reference (agents alive in both)

| variant | closed | clamped | blind | closed − clamped |
|---|---|---|---|---|
| sw_full | +0.143 [-0.188, +0.475] | -0.074 [-0.208, +0.061] | -0.028 [-0.576, +0.519] | +0.186 [-0.174, +0.546] |
| sw_pe | -0.153 [-0.276, -0.030] | -0.242 [-0.510, +0.026] | -0.067 [-0.306, +0.173] | +0.087 [-0.257, +0.432] |
| sw_pecost | -0.028 [-0.564, +0.508] | -0.068 [-0.252, +0.115] | +0.079 [-0.359, +0.516] | +0.037 [-0.259, +0.333] |
| sw_pecost_pe | -0.022 [-0.638, +0.593] | -0.154 [-0.373, +0.065] | -0.084 [-0.578, +0.411] | +0.178 [-0.768, +1.123] |
| pecost_pe_static | +0.037 [-0.333, +0.407] | -0.064 [-0.500, +0.372] | +0.047 [+0.012, +0.083] | +0.060 [-0.572, +0.691] |
| sw_pecost_pe_shuffled | +0.072 [-0.229, +0.372] | -0.059 [-0.231, +0.113] | -0.173 [-0.609, +0.262] | +0.123 [-0.150, +0.396] |
| sw_no_cost | -0.014 [-0.124, +0.096] | +0.008 [-0.004, +0.020] | +0.017 [-0.063, +0.098] | -0.022 [-0.140, +0.096] |

## Hidden-state kicks scored on regulated variables vs world-chaos floor (mean |Δ| over steps 60–120 vs unperturbed reference)

| variant | |ΔE| kick 0.1 | |ΔE| kick 0.3 | |ΔE| twin (new weather) | log(kick0.1/twin) | log(kick0.3/twin) | |ΔE| early (0–20) kick0.1 / twin | |Δintake| kick0.1 / twin | |ΔPE| kick0.1 / twin |
|---|---|---|---|---|---|---|---|---|
| sw_full | +0.152 [-0.012, +0.316] | +0.191 [+0.000, +0.382] | +0.291 [+0.194, +0.388] | -2.315 [-5.966, +1.335] | -1.376 [-4.100, +1.349] | 0.004 / 0.036 | 0.188 / 0.428 | 0.5081 / 0.7884 |
| sw_pe | +0.160 [-0.019, +0.339] | +0.190 [+0.016, +0.364] | +0.281 [+0.239, +0.324] | -1.773 [-5.089, +1.543] | -1.199 [-3.333, +0.934] | 0.007 / 0.055 | 0.161 / 0.453 | 0.4560 / 0.7107 |
| sw_pecost | +0.104 [-0.014, +0.223] | +0.191 [+0.037, +0.345] | +0.330 [+0.193, +0.466] | -4.036 [-7.393, -0.678] | -1.386 [-2.699, -0.073] | 0.002 / 0.054 | 0.099 / 0.415 | 0.2477 / 0.7267 |
| sw_pecost_pe | +0.181 [+0.040, +0.321] | +0.229 [+0.077, +0.382] | +0.358 [+0.305, +0.412] | -2.327 [-5.152, +0.498] | -1.347 [-3.531, +0.837] | 0.005 / 0.043 | 0.216 / 0.535 | 0.4508 / 0.6960 |
| pecost_pe_static | +0.090 [+0.027, +0.153] | +0.203 [+0.060, +0.345] | +0.364 [+0.193, +0.535] | -3.369 [-5.394, -1.345] | -1.280 [-2.775, +0.215] | 0.005 / 0.030 | 0.096 / 0.392 | 0.2803 / 0.7021 |
| sw_pecost_pe_shuffled | +0.148 [+0.012, +0.284] | +0.210 [+0.089, +0.332] | +0.327 [+0.283, +0.371] | -2.501 [-6.634, +1.633] | -1.177 [-3.463, +1.109] | 0.005 / 0.048 | 0.148 / 0.470 | 0.6877 / 0.9909 |
| sw_no_cost | +0.090 [+0.071, +0.109] | +0.097 [+0.059, +0.135] | +0.100 [+0.051, +0.149] | -0.171 [-0.293, -0.048] | -0.134 [-0.329, +0.062] | 0.005 / 0.002 | 0.074 / 0.094 | 0.8514 / 0.9343 |

## Verdict (active recovery requires the 95% CI of mean A to exclude 0 on the positive side AND the CI of P(closed deficit < clamped deficit) to exclude 0.5)

| variant | energy recovery | via sensing (B > 0) | kick effect on E exceeds world noise |
|---|---|---|---|
| sw_full | no detectable difference | False | False |
| sw_pe | no detectable difference | False | False |
| sw_pecost | passive ≥ active | False | False |
| sw_pecost_pe | passive ≥ active | False | False |
| pecost_pe_static | no detectable difference | False | False |
| sw_pecost_pe_shuffled | no detectable difference | False | False |
| sw_no_cost | no detectable difference | False | False |

## Validation

Clamping an unperturbed target to its own reference actions reproduces the reference energy exactly while the reference agent lives (targets whose reference survives 120 steps; max |ΔE| = 0.00e+00). Over all targets it is 1.46e+00, because the recorded action is zeroed on the step the reference agent dies; that is why energy-drop metrics use only targets whose reference survives.
Targets used per run (reference alive at step 120): 575 of 757.
