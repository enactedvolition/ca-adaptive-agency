"""Exp 24: sim13 ctl["tuse"]/ctl["tmap"]: (a) ctl without the keys == ctl with tuse all False (any tmap) == ctl=None, bit-for-bit;
(b) tuse True: logged thrust == tanh(x . tmap) (x = the step's sensed inputs, xprev), other channels brain-chosen (== the unhooked run's first step)."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp22_run as X
FORK = os.path.join(X.HERE, "runs20", "fork20_s0_t1500.npz")

@pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")
def test_tmap_hook():
    import jax, jax.numpy as jnp, exp19_run as X19
    cfg, chunk_one, consts, prm, tmpl, off = X19.world(); step = consts["step"]; N = cfg.N
    z = np.load(FORK); s = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
    rng = np.random.default_rng(0); tm = jnp.asarray(rng.normal(0, 0.5, (N, 16, 2)), jnp.float32); tu = jnp.asarray(rng.random(N) < 0.3)
    base = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N))
    f = jax.jit(lambda c, ctl: step(c, prm, ctl)); f0 = jax.jit(lambda c: step(c, prm, None))
    eq = lambda a, b: all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in a)
    a0, aN, aF = s, s, s
    for _ in range(8):
        a0, aN, aF = f0(a0), f(aN, base), f(aF, dict(base, tuse=jnp.zeros(N, bool), tmap=tm))
    assert eq(aN, a0) and eq(aF, a0)
    o = f(s, dict(base, tuse=tu, tmap=tm)); o0 = f0(s)
    act, x, al = np.asarray(o["act"]), np.asarray(o["xprev"]), np.asarray(o["alive"]); c = np.asarray(tu) & al
    pred = np.tanh(np.einsum("ni,nic->nc", x, np.asarray(tm)))
    assert np.allclose(act[c][:, :2], pred[c], atol=1e-6, rtol=0)
    assert np.array_equal(act[c][:, 2:], np.asarray(o0["act"])[c][:, 2:])      # same pre-step state -> same brain output on other channels
    assert np.array_equal(x, np.asarray(o0["xprev"]))                           # senses are pre-action, unaffected by the hook
    assert not np.allclose(act[c][:, :2], np.asarray(o0["act"])[c][:, :2])
