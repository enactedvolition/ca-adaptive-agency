"""Exp 21 (no-implant control): clamp0 (OPEN clamp, no drop) reproduces ref exactly; FROZEN with no drop also equals ref (e_off = 0 when E == E_ref); targets valid."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp21_run as X

@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(X.HERE), "mvp", "runs17", "fork16_sw_pays_s0_t3000.npz")), reason="Exp 17 fork not present")
def test_clamp0_equals_ref_and_targets():
    import jax.numpy as jnp
    f, base, tg, K, off = X.build(0); bj = {k: jnp.asarray(v) for k, v in base.items()}; n = 20
    z8 = jnp.zeros((n, K, 8)); ze = jnp.zeros((n, K)); F, Tr = jnp.bool_(False), jnp.bool_(True); key = jnp.asarray(base["key"])
    ref = {k: np.asarray(v) for k, v in f(bj, key, 0.0, F, F, z8, ze).items()}
    c0 = {k: np.asarray(v) for k, v in f(bj, key, 0.0, Tr, F, jnp.asarray(ref["act"]), ze).items()}
    assert np.array_equal(c0["E"], ref["E"]) and np.array_equal(c0["passed"], ref["passed"]) and np.array_equal(c0["births"], ref["births"])
    eref = np.where(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0), np.concatenate([base["E"][tg][None], ref["E"][:-1]], 0), -1.0).astype(np.float32)
    fr0 = {k: np.asarray(v) for k, v in f(bj, key, 0.0, F, Tr, z8, jnp.asarray(eref)).items()}
    assert np.abs(fr0["E"] - ref["E"]).max() < 1e-5          # frozen with no drop: sensed E == own E up to float rounding of the offset
    assert len(tg) == K == 40 and (base["E"][tg] >= X.E_MIN).all() and base["alive"][tg].all() and len(np.unique(tg)) == K
    assert (ref["passed"][ref["births"] == 0] == 0).all() and (ref["passed"] <= 0.6335 + 1e-4).all()
