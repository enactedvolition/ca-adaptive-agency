"""sim13 with rep_fixed = 0 must reproduce sim12 bit for bit (Exp 17); with rep_fixed = 1 it must differ and never create energy."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mvp"))

def _run(mod, steps=40, **kw):
    import jax
    cfg = mod.Cfg(N=256, L=1.0, G=32, claim_mode="mass")
    init, chunk_one, colour, consts = mod.make(cfg)
    prm = mod.params(c_basal=0.004, mut_sigma=0.05, env_var=0.35, sw_period=300.0, **kw)
    s = init(jax.random.PRNGKey(7), prm)
    return chunk_one(s, prm, steps)

def test_sim13_off_equals_sim12():
    import sim12, sim13
    for kw in (dict(), dict(rep_free=1.0)):
        a = _run(sim12, **kw); b = _run(sim13, **kw)
        assert set(a) == set(b)
        bad = [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True)]
        assert not bad, bad

def test_sim13_on_differs():
    import sim13
    a = _run(sim13, rep_free=1.0); b = _run(sim13, rep_free=1.0, rep_fixed=1.0)
    assert not np.array_equal(np.asarray(a["E"]), np.asarray(b["E"]))
    E = np.asarray(b["E"])[np.asarray(b["alive"])]
    assert (E > 0).all()
