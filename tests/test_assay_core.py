"""Rebuild step 0 (pre-registered NIGHTLY_LOG 2026-11-28 04:49 PST, commit 6e78c54): e_fix hook + assay_core identity (no-change) checks on a real fork.
The Exp 20 fork is world background only; nothing here is a result about any population."""
import os, sys, subprocess, importlib.util, tempfile
from dataclasses import asdict
import numpy as np, pytest
HERE = os.path.join(os.path.dirname(__file__), "..", "mvp"); sys.path.insert(0, HERE)
FORK = os.path.join(HERE, "runs20", "fork20_s0_t1500.npz")
REF = "153c43b"   # sim13 before the e_fix hook
T, K = 16, 64
pytestmark = pytest.mark.skipif(not os.path.exists(FORK), reason="Exp 20 fork not present")

def _old_sim13():
    try:
        src = subprocess.check_output(["git", "-C", HERE, "show", f"{REF}:mvp/sim13.py"], text=True)
    except Exception:
        pytest.skip("git ref not available")
    p = os.path.join(tempfile.mkdtemp(), "sim13_pre_efix.py"); open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("sim13_pre_efix", p); m = importlib.util.module_from_spec(spec)
    sys.modules["sim13_pre_efix"] = m; spec.loader.exec_module(m); return m

@pytest.fixture(scope="module")
def W():
    import jax, jax.numpy as jnp, exp19_run as X19
    cfg, chunk_one, consts, prm, tmpl, off = X19.world()
    z = np.load(FORK); s = {k: jnp.asarray(np.asarray(z[k]).astype(sd.dtype)) for k, sd in tmpl.items()}
    elig = np.nonzero(np.asarray(s["alive"]) & (np.asarray(s["E"]) >= 0.3))[0]
    tg = np.sort(np.random.default_rng(0).choice(elig, K, replace=False))
    return dict(jax=jax, jnp=jnp, cfg=cfg, step=consts["step"], prm=prm, s=s, N=cfg.N, tg=tg)

def test_efix_absent_equals_pre_change_sim13(W):
    jax, jnp, N, step, prm, s = W["jax"], W["jnp"], W["N"], W["step"], W["prm"], W["s"]
    old = _old_sim13(); _, _, _, oc = old.make(old.Cfg(**asdict(W["cfg"]))); ostep = oc["step"]
    base = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N), rep_use=jnp.ones(N, bool), rep_over=jnp.full((N,), -1.0))
    off = dict(base, e_fix_use=jnp.zeros(N, bool), e_fix=jnp.full((N,), 3.0))
    eq = lambda a, b: all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in a)
    fs = [jax.jit(lambda c: step(c, prm, None)), jax.jit(lambda c: ostep(c, prm, None)),
          jax.jit(lambda c: step(c, prm, base)), jax.jit(lambda c: ostep(c, prm, base)), jax.jit(lambda c: step(c, prm, off))]
    st = [s] * 5
    for _ in range(8):
        st = [f(x) for f, x in zip(fs, st)]
    assert eq(st[0], st[1])      # ctl=None: new == pre-change
    assert eq(st[2], st[3])      # Exp 22-style ctl without e_fix: new == pre-change
    assert eq(st[4], st[3])      # e_fix_use all False == no e_fix keys

def test_efix_sets_channel_exactly_and_acts(W):
    jax, jnp, N, step, prm, s, tg = W["jax"], W["jnp"], W["N"], W["step"], W["prm"], W["s"], W["tg"]
    use = jnp.zeros(N, bool).at[tg].set(True); val = jnp.zeros(N).at[tg].set(jnp.float32(-0.55))
    base = dict(aclamp=jnp.zeros(N, bool), aover=jnp.zeros((N, 8)), e_off=jnp.zeros(N))
    o = step(s, prm, dict(base, e_fix_use=use, e_fix=val)); r = step(s, prm, base)
    x8 = np.asarray(o["xprev"])[:, 8]; xr = np.asarray(r["xprev"])[:, 8]; m = np.zeros(N, bool); m[tg] = True
    assert np.all(x8[tg] == np.float32(-0.55))           # exact on targets
    assert np.array_equal(x8[~m], xr[~m])                 # untouched elsewhere (same step, same state)
    assert not np.array_equal(np.asarray(o["act"])[tg], np.asarray(r["act"])[tg])   # a sighted brain responds to its sense

@pytest.fixture(scope="module")
def H(W):
    import assay_core as A
    run = A.build(W["step"], W["prm"], W["N"], W["tg"], T)
    ref = run(W["s"], "ref"); return dict(A=A, run=run, ref=ref)

def test_identities_brain(H, W):
    A, run, ref, s = H["A"], H["run"], H["ref"], W["s"]
    assert ref["alive"][0].all() and ref["alive"][-1].sum() > 0
    assert A.identical(run(s, "open", ref=ref, drop=0.0), ref), "clamp0 (OPEN without drop) != ref"
    assert A.identical(run(s, "frozen", ref=ref, drop=0.0), ref), "frozen0 (sense-level replay without drop) != ref"
    assert A.identical(run(s, "fixown"), ref), "fixown (e_fix = own current sensed value) != ref"
    cl = run(s, "cl", drop=0.5)
    assert not A.identical(cl, ref)                      # the drop does something
    fr = run(s, "frozen", ref=ref, drop=0.5)
    assert np.array_equal(fr["x8"][fr["alive"]], ref["x8"][fr["alive"]])   # FROZEN targets sensed exactly what ref sensed
    assert not np.array_equal(cl["x8"][:1], ref["x8"][:1])                 # ...while CL targets sensed the drop at step 0

def test_deliberate_failure_post_step_lead_is_caught(H, W):
    """Built to FAIL the identity: a FROZEN arm fed the Exp 25/26 convention (ref POST-step energy, rendered as a sense value at the
    same step). The no-change check must reject it; if this ever passes as identical, the checker is blind to the known bug."""
    A, run, ref, s = H["A"], H["run"], H["ref"], W["s"]
    bad = run(s, "frozen", ref=ref, drop=0.0, xrec=A.post_step_lead_x8(ref))
    assert not A.identical(bad, ref)
    assert not np.array_equal(A.post_step_lead_x8(ref)[0], ref["x8"][0])

def test_identities_controller_blind_haz_birthsoff(W):
    """External STEP eat controller (Exp 27 style: brains blind, births off, predation on): clamp0 / frozen0 == ref, lead caught."""
    import assay_core as A
    jnp = W["jnp"]; prm = dict(W["prm"]); prm["blind"] = jnp.float32(1.0)
    pol = lambda u, c: (u < 0.40).astype(jnp.float32)
    run = A.build(W["step"], prm, W["N"], W["tg"], T, rep_off="all", haz=0.004, policy=pol); s = W["s"]
    ref = run(s, "ref")
    assert ref["births"].sum() == 0 and np.array_equal(ref["u"][0], np.asarray(s["E"])[W["tg"]])
    assert A.identical(run(s, "open", ref=ref, drop=0.0), ref, keys=("E", "alive", "act", "eat"))
    assert A.identical(run(s, "frozen", ref=ref, drop=0.0), ref, keys=("E", "alive", "act", "eat"))
    lead = np.where(ref["alive"], ref["E"], 0.0).astype(np.float32)
    assert not A.identical(run(s, "frozen", ref=ref, drop=0.0, urec=lead), ref, keys=("E", "alive", "act", "eat"))
    fr = run(s, "frozen", ref=ref, drop=0.5); cl = run(s, "cl", drop=0.5)
    m = fr["alive"]; assert np.array_equal(fr["u"][m], ref["u"][m]) and not np.array_equal(cl["u"][0], ref["u"][0])
