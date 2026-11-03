# Nightly research log

Written by the nightly research routine. Newest entries at the bottom.

## 2026-11-03

- 01:20 PST: Exp 4's planned follow-up was not done yet, so tonight runs it as Exp 5: charge energy for prediction error itself (0.01 × the agent's own next-input predictor MSE per step) in a world whose food layout switches every 300 steps. Seven variants × 3 seeds (sw_full, sw_pe, sw_pecost, sw_pecost_pe, pecost_pe_static, sw_pecost_pe_shuffled, sw_no_cost), then the Price decomposition and perturbation recovery (closed vs open-loop vs frozen-input), judged with the same decision rule. Checked that `sim5` with both new knobs off reproduces `sim4` bit for bit. The venv had disappeared, so it was rebuilt from the pinned requirements. Running detached as `mvp/run5_chain.sh`; it resumes from per-file checkpoints and commits only if validation and the smoke test pass.

- 2026-11-03 01:49 PST: Exp 5 (prediction-error cost in a switching world) finished. No variant shows active recovery beating passive replay; every one gets the same passive label as Exp 4. Main arm sw_pecost_pe closed−open kick0.1 = +17.77 [+13.28, +22.27]; label: stable inherited dynamics. Committed to the public repo (README section, figures/exp5).
