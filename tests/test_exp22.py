"""Exp 22 (births off for targets): (a) ctl=None step == ctl step with zero clamp / zero e_off / rep_use all False, bit-for-bit;
(b) the Exp 22 build with rep_off=False reproduces the stored Exp 20 seed-0 ref trajectory exactly (so the only change is the rep override);
(c) with rep off: targets make 0 births, clamp0 == ref exactly, rep override is not a no-op (ref differs from the rep-on ref)."""
import os, sys, pickle
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))
import exp22_run as X
FORK = os.path.join(X.HERE, "runs20", "fork20_s0_t1500.npz"); P20 = os.path.join(X.HERE, "runs20", "run20_s0.pkl")

def test_ctl_none_bit_identical():
    import jax, jax.numpy as jnp, exp19_run as X19
    cfg, chunk_one, consts, prm, tmpl, off = X19.world(); step = consts["step"]; N = cfg.N
    if not os.path.exists(FORK): pytest.skip("Exp 20 fork not present")
    z = np.load(FORK); s = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
    ctl = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.zeros(N, bool), rep_over=jnp.full((N,), -1.0))
    f0 = jax.jit(lambda c: step(c, prm, None)); f1 = jax.jit(lambda c: step(c, prm, ctl))
    a, b = s, s
    for _ in range(10):
        a, b = f0(a), f1(b)
    for k in a: assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k   # prate is NaN for agents with no record

@pytest.mark.skipif(not (os.path.exists(FORK) and os.path.exists(P20)), reason="Exp 20 fork/run not present")
def test_rep_off_false_reproduces_exp20_and_rep_off_valid():
    import jax.numpy as jnp
    d20 = pickle.load(open(P20, "rb")); F, Tr = jnp.bool_(False), jnp.bool_(True)
    f, base, tg, K, off = X.build("true", 0, rep_off=False); assert np.array_equal(tg, d20["targets"])
    bj = {k: jnp.asarray(v) for k, v in base.items()}; key = jnp.asarray(base["key"]); z8 = jnp.zeros((X.T, K, 8)); ze = jnp.zeros((X.T, K))
    r_on = {k: np.asarray(v) for k, v in f(bj, key, 0.0, F, F, z8, ze).items()}
    assert np.array_equal(r_on["E"], d20["arms"]["ref"]["E"]) and np.array_equal(r_on["births"], d20["arms"]["ref"]["births"])
    f, base, tg, K, off = X.build("true", 0, rep_off=True)
    ref = {k: np.asarray(v) for k, v in f(bj, key, 0.0, F, F, z8, ze).items()}
    c0 = {k: np.asarray(v) for k, v in f(bj, key, 0.0, Tr, F, jnp.asarray(ref["act"]), ze).items()}
    cl = {k: np.asarray(v) for k, v in f(bj, key, 1.0, F, F, z8, ze).items()}
    assert ref["births"].sum() == 0 and c0["births"].sum() == 0 and cl["births"].sum() == 0 and ref["passed"].sum() == 0
    assert np.array_equal(c0["E"], ref["E"])
    assert r_on["births"].sum() > 0 and not np.array_equal(r_on["E"], ref["E"])
