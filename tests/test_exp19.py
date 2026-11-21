"""Exp 19 implant: rewired brains reproduce iff sensed E > 0.8 (surviving agents), and the ctl path with e_off = 0 equals plain sim13."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))

def test_implant_rule_and_ctl_path():
    import jax, jax.numpy as jnp, sim13, exp19_run as X
    cfg = sim13.Cfg(N=256, L=1.0, G=32, claim_mode="mass"); init, chunk_one, colour, consts = sim13.make(cfg)
    prm = sim13.params(**X.BASE); off = {n: (o, sh) for n, sh, o, z in consts["lay"]}
    s = chunk_one(init(jax.random.PRNGKey(3), prm), prm, 20); s = {k: np.asarray(v) for k, v in s.items()}
    s["E"] = np.where(s["alive"], np.random.default_rng(0).uniform(0.4, 1.6, s["E"].shape), 0.0).astype(np.float32)   # spread E around THR
    s, m = X.implant(s, 0, off); sel = m & s["alive"]; assert sel.sum() > 10
    J = {k: jnp.asarray(v) for k, v in s.items()}; N = cfg.N
    ctl = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.zeros(N, bool), rep_over=jnp.zeros(N))
    a = consts["step"](J, prm, ctl); b = consts["step"](J, prm, None)
    assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in a)
    rp = np.asarray(a["act"][:, 3]) > 0; surv = sel & np.asarray(a["alive"])
    assert np.array_equal(rp[surv], s["E"][surv] > X.THR)
