"""Exp 20: recovery metric arithmetic, and (if the t=1500 fork exists) clamp0 == ref exactly, passed-energy bookkeeping == births * child E."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp20_run as X

def test_metric_arithmetic():
    T, K = 10, 2; E0 = np.array([1.0, 0.8])
    ref_E = np.full((T, K), 1.0); ref_P = np.zeros((T, K)); ref_P[2, 0] = 0.6; ref_E[2:, 0] -= 0.8   # ref target 0 gives birth at step 2 (parent pays 0.8, child gets 0.6)
    assert np.allclose(X.lineage_L(ref_E, ref_P)[-1], [0.2 + 0.6, 1.0])
    r = X.recovery(ref_E, ref_P, ref_E, ref_P, E0, lo=5); assert np.allclose(r, 1.0)               # identical to ref: full recovery
    gone = np.zeros((T, K)); r = X.recovery(gone, np.zeros((T, K)), ref_E, ref_P, E0, lo=5)       # dead, no births: deficit = L_ref
    assert np.allclose(r, [1 - 0.8 / 0.5, 1 - 1.0 / 0.4])
    dE = X.DROP * E0; arm_E = ref_E - dE; r = X.recovery(arm_E, ref_P, ref_E, ref_P, E0, lo=5); assert np.allclose(r, 0.0)   # deficit == removed amount

@pytest.mark.skipif(not os.path.exists(os.path.join(X.OUT, "fork20_s0_t1500.npz")), reason="fork not generated")
def test_clamp_and_bookkeeping():
    import jax.numpy as jnp
    f, base, tg, K, off = X.build(0); bj = {k: jnp.asarray(v) for k, v in base.items()}; n = 20
    z8 = jnp.zeros((n, K, 8)); ze = jnp.zeros((n, K)); F, Tr = jnp.bool_(False), jnp.bool_(True)
    ref = {k: np.asarray(v) for k, v in f(bj, jnp.asarray(base["key"]), 0.0, F, F, z8, ze).items()}
    c0 = {k: np.asarray(v) for k, v in f(bj, jnp.asarray(base["key"]), 0.0, Tr, F, jnp.asarray(ref["act"]), ze).items()}
    assert np.array_equal(c0["E"], ref["E"]) and np.array_equal(c0["passed"], ref["passed"])
    assert (ref["passed"][ref["births"] == 0] == 0).all()
    assert (ref["passed"][ref["births"] > 0] <= 0.6335 + 1e-4).all()
    assert len(tg) == K and (base["E"][tg] >= X.E_MIN).all() and base["alive"][tg].all()
