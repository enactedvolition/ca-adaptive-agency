"""Exp 25: sim13 predation hook ctl["haz"]/ctl["hu"]. Off path == pre-change sim13 (git 14a122f) bit for bit; on path kills exactly hu < haz*eat."""
import os, sys, subprocess, importlib.util, tempfile
import numpy as np, pytest
HERE = os.path.join(os.path.dirname(__file__), "..", "mvp"); sys.path.insert(0, HERE)
FORK = os.path.join(HERE, "runs20", "fork20_s0_t1500.npz")
REF = "14a122f"

def _old_sim13():
    try:
        src = subprocess.check_output(["git", "-C", HERE, "show", f"{REF}:mvp/sim13.py"], text=True)
    except Exception:
        pytest.skip("git ref not available")
    p = os.path.join(tempfile.mkdtemp(), "sim13_old.py"); open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("sim13_old", p); m = importlib.util.module_from_spec(spec); sys.modules["sim13_old"] = m; spec.loader.exec_module(m); return m

@pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")
def test_haz_hook():
    import jax, jax.numpy as jnp, exp19_run as X19
    cfg, chunk_one, consts, prm, tmpl, off = X19.world(); step = consts["step"]; N = cfg.N
    old = _old_sim13(); from dataclasses import asdict; _, _, _, oc = old.make(old.Cfg(**asdict(cfg))); ostep = oc["step"]
    z = np.load(FORK); s = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
    base = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0))
    hu = jnp.asarray(np.random.default_rng(0).random(N), jnp.float32)
    eq = lambda a, b: all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in a)
    fN = jax.jit(lambda c: step(c, prm, None)); fo = jax.jit(lambda c: ostep(c, prm, None))
    fB = jax.jit(lambda c: step(c, prm, base)); foB = jax.jit(lambda c: ostep(c, prm, base))
    fZ = jax.jit(lambda c: step(c, prm, dict(base, haz=jnp.float32(0.0), hu=hu)))
    a, b, c, d, e = s, s, s, s, s
    for _ in range(8):
        a, b, c, d, e = fN(a), fo(b), fB(c), foB(d), fZ(e)
    assert eq(a, b)            # ctl=None: new == old sim13
    assert eq(c, d)            # Exp 22-style ctl without haz: new == old
    assert eq(e, d)            # haz = 0: == old
    # on path: one step, haz = 0.5
    o = jax.jit(lambda c: step(c, prm, dict(base, haz=jnp.float32(0.5), hu=hu)))(s); o0 = fB(s)
    eat = (np.asarray(o0["act"])[:, 2] + 1) / 2; al0 = np.asarray(o0["alive"]); kill = np.asarray(s["alive"]) & (np.asarray(hu) < 0.5 * eat)
    assert kill.sum() > 10
    assert np.array_equal(np.asarray(o["alive"])[~kill], al0[~kill]) and not np.asarray(o["alive"])[kill].any()
