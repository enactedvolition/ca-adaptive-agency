"""Exp 23: sim13 ctl["amask"]: (a) amask all-False with aclamp True == ctl=None; (b) amask all-True == the Exp 22 full clamp (no amask), bit-for-bit;
(c) partial mask replays only the masked channels (other channels stay brain-chosen)."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp22_run as X
FORK = os.path.join(X.HERE, "runs20", "fork20_s0_t1500.npz")

@pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")
def test_amask():
    import jax, jax.numpy as jnp, exp19_run as X19
    cfg, chunk_one, consts, prm, tmpl, off = X19.world(); step = consts["step"]; N = cfg.N
    z = np.load(FORK); s = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
    rng = np.random.default_rng(0); aov = jnp.asarray(rng.uniform(-1, 1, (N, 8)), jnp.float32); cl = jnp.asarray(rng.random(N) < 0.3)
    base = dict(aclamp=cl, aover=aov, e_off=jnp.zeros(N))
    mk = lambda m: dict(base, amask=jnp.broadcast_to(jnp.asarray(m, bool), (N, 8)))
    f = jax.jit(lambda c, ctl: step(c, prm, ctl)); f0 = jax.jit(lambda c: step(c, prm, None))
    eq = lambda a, b: all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in a)
    a0, aT, aN, aF = s, s, s, s
    for _ in range(8):
        a0, aT, aN, aF = f0(a0), f(aT, mk([1] * 8)), f(aN, base), f(aF, mk([0] * 8))
    assert eq(aT, aN)                      # all-True mask == existing full clamp
    # all-False mask == no clamp at all (ctl without clamp)
    nc = dict(base, aclamp=jnp.zeros(N, bool)); b = s
    for _ in range(8): b = f(b, nc)
    assert eq(aF, b) and eq(b, a0)
    # partial mask: act of clamped agents on masked channels equals aov, others are not aov
    m = [1, 1, 0, 0, 0, 0, 0, 0]; o = f(s, mk(m)); act = np.asarray(o["act"]); c = np.asarray(cl) & np.asarray(o["alive"])
    assert np.array_equal(act[c][:, :2], np.asarray(aov)[c][:, :2]) and not np.allclose(act[c][:, 2:], np.asarray(aov)[c][:, 2:])
