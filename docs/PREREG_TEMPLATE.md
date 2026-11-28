# Pre-registration template (rebuild steps; append to NIGHTLY_LOG.md and commit BEFORE running anything)

- **Step / title:** (REBUILD_PLAN step number and name)
- **Timestamp (PST) and code hash** that will run (commit the code first, or state "code written after this entry, hash in result entry").
- **Question and level:** the one question, at a named level L0–L4 (EPISTEMOLOGY.md §1) and rung R0–R7.
- **Subjects:** which systems are tested. Constructed systems = calibration only; say so.
- **Design-seed ledger:** every seed/fork/key that informed any design choice so far (excluded from confirmatory tests).
- **Arms and timing contract:** each arm, what state it reads at step t, what it writes; use `mvp/assay_core.py`.
- **Identity (null-manipulation) tests** for every arm, bit for bit, run before the real contrast.
- **Primary contrast:** one statistic, its unit of replication (seed/fork), its test, α.
- **Absence claims:** TOST bound δ in calibrated units (from a ladder), or "no absence claim will be made".
- **Gate:** pass/fail criteria, written as numbers.
- **What a fail means** (informative vs uninformative, EPISTEMOLOGY.md §4) and what happens next.
- **Compute plan:** jobs, detached runner, checkpoint path, expected wall time.
- **Deviations:** filled in only in the result entry, each marked as a deviation.
