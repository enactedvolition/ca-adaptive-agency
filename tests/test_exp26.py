"""Exp 26: target picker (K=200 distinct valid), survival function, and clamp0 == ref on a real fork (12 steps)."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp26_pop as X
FORK = os.path.join(X.HERE, "runs20", "fork20_s0_t1500.npz")

def test_survival_fn():
    al = np.zeros((300, 4), bool); al[:150, :3] = True; al[:100, 3] = True
    assert X.survival(al, 150) == 0.75 and X.survival(al, 100) == 1.0 and X.survival(al, 151) == 0.0

@pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")
def test_picker_and_clamp0():
    import jax, jax.numpy as jnp
    f, base, tg, K, off = X.build("true", 0)
    assert K == 200 and len(set(tg.tolist())) == 200 and base["alive"][tg].all() and (base["E"][tg] >= X.E_MIN).all()
    assert np.array_equal(tg, X.pick_targets("true", base, off, 0))
    bj = {k: jnp.asarray(v) for k, v in base.items()}; key0 = jnp.asarray(base["key"]); z8 = jnp.zeros((X.T, K, 8)); ze = jnp.zeros((X.T, K)); tz = jnp.zeros((K, X.NX, 2))
    F, Tr = jnp.bool_(False), jnp.bool_(True)
    ref = {k: np.asarray(v) for k, v in f(bj, key0, 0.0, F, F, z8, ze, F, tz, nsteps=12).items()}
    c0 = {k: np.asarray(v) for k, v in f(bj, key0, 0.0, Tr, F, jnp.zeros((X.T, K, 8)).at[:12].set(jnp.asarray(ref["act"])), ze, F, tz, nsteps=12).items()}
    assert np.array_equal(c0["E"], ref["E"]) and np.array_equal(c0["alive"], ref["alive"]) and ref["births"].sum() == 0
