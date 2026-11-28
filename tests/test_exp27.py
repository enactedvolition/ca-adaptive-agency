"""Exp 27: controller family (numpy mirror), and on a real fork: clamp0 == ref bit for bit, 0 births, and FROZEN == CL exactly for an
input-ignoring controller (P0), FROZEN != CL for an energy-feedback controller (STEP)."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp27_run as X
FORK = os.path.join(X.HERE, "runs20", "fork20_s0_t1500.npz")

def test_policy_family():
    u = np.array([0.0, 0.2, 0.4, 0.6, 2.0])
    assert np.allclose(X.policy_np(X.spec(0, 0.0), u, 0), 0.5)
    assert np.allclose(X.policy_np(X.spec(0, 1.0), u, 0), [0.9, 0.7, 0.5, 0.3, 0.0])
    assert np.array_equal(X.policy_np(X.spec(1), u, 0), [1, 1, 0, 0, 0])
    w = X.w_init(); w[24:32] = 1.0
    a = X.policy_np(X.spec(2, src=2, w=w), u, 0.3); assert np.allclose(a, a[0])          # src 2 ignores the energy input
    assert not np.allclose(X.policy_np(X.spec(2, w=w), u, 0.3), a[0])

@pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")
def test_validity_on_fork():
    r0 = X.assay_job(("P0", X.spec(0, 0.0), 0, 0, X.HAZ), nsteps=True)
    assert r0["clamp0_exact"] and r0["frozen_eq_cl"] and r0["births"] == 0 and r0["G_eff"] == 0.0
    r1 = X.assay_job(("STEP", X.spec(1), 0, 0, X.HAZ), nsteps=True)
    assert r1["clamp0_exact"] and not r1["frozen_eq_cl"] and r1["births"] == 0
    assert len(set(r1["tg"].tolist())) == X.K_T and (r1["E0"] >= X.E_MIN).all()
