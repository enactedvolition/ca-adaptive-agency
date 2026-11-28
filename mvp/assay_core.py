"""Rebuild step 0 (docs/REBUILD_PLAN.md; pre-registered in NIGHTLY_LOG 2026-11-28): shared perturbation-assay harness for sim13 worlds.

One jitted scan serves every arm. Timing contract (step t = one call of sim13 step):
  * every arm reads the state at the START of step t (pre-step E, alive, uid);
  * the record of step t holds what the agent SENSED and DID inside step t (`x8` = c2["xprev"][:, 8], the brain's channel-8 input
    actually used; `u` = an external eat controller's input actually used; `act` = c2["act"]) and its state AFTER step t (`E`, `alive`);
  * replay arms feed record[t] back at step t. FROZEN replays the recorded SENSED input (x8 via sim13 ctl e_fix, u for a controller);
    it never reconstructs a sensed value from stored energy (the Exp 25/26 post-step-lead bug cannot be expressed through `run_arm`).
Arms: ref (no drop), cl (drop), open (drop; all 8 actions, and controller eat, replayed from ref while the ref target's recorded action
is non-zero), frozen (drop; sensed input replayed while the ref target was alive at the start of the step), ablate (drop; sensed input
fixed to a given constant), fixown (no drop; e_fix = the agent's own current sensed value: identity test of the hook).
Identity (no-change) checks: clamp0 = open w/o drop, frozen0 = frozen w/o drop, fixown == ref, bit for bit (identical()).
Targets = indices `tg`; `uid0` taken from the start state; a slot reassigned to another uid is no longer a target."""
import numpy as np

REC_KEYS = ("E", "alive", "act", "x8", "u", "eat", "births")

def build(step, prm, N, tg, T, rep_off="none", haz=None, policy=None, brain_sense=True):
    """step/prm: sim13 step and params. rep_off: 'none' | 'targets' | 'all' (rep forced -1, same in every arm).
    haz: None or float (sim13 predation hook; uniforms drawn from the hazard key, identical across arms).
    brain_sense: if False, the ablate/sscale arms act on the controller input u only (brains untouched; FROZEN still replays x8).
    policy: None or f(u (N,), c, pp) -> eat (N,) in [0,1] (pp = run_arm(..., pp=) parameter array), an external eat controller for every live cohort agent (input u = own pre-step E
    unless replayed/fixed for targets). Returns run_arm(s, arm, ref=None, drop=0.5, key=None, hkey=None, fix=None, xrec=None, urec=None)."""
    import jax, jax.numpy as jnp
    tg = np.asarray(tg); K = len(tg); tg_j = jnp.asarray(tg, jnp.int32)

    def core(s, key, hkey, drop, clamp, frozen, fixc, fixown, sscale, aov, eov, xrec, urec, xval, fixv, pp):
        s = dict(s); s["key"] = key; uid0 = s["uid"]; coh = s["alive"]
        s["E"] = s["E"].at[tg_j].add(-drop * s["E"][tg_j])
        def body(c, xs):
            hk, a_t, ea_t, x_t, u_t, v_t = xs
            sameN = c["alive"] & coh & (c["uid"] == uid0); same = sameN[tg_j]
            cl_T = clamp & same & (jnp.abs(a_t).sum(1) > 0)
            fr_T = frozen & same & v_t
            fx_T = (fixown | ((fixc | sscale) & brain_sense)) & same
            xfix = jnp.where(fixown, jnp.log1p(c["E"][tg_j]) - 0.7, jnp.where(sscale, jnp.log1p(fixv * c["E"][tg_j]) - 0.7, fixv))
            efix = jnp.where(fr_T, x_t, xfix)
            ctl = dict(aclamp=jnp.zeros(N, bool).at[tg_j].set(cl_T), aover=jnp.zeros((N, 8)).at[tg_j].set(a_t), e_off=jnp.zeros(N),
                       e_fix_use=jnp.zeros(N, bool).at[tg_j].set(fr_T | fx_T), e_fix=jnp.zeros(N).at[tg_j].set(efix))
            if rep_off != "none":
                ru = jnp.ones(N, bool) if rep_off == "all" else jnp.zeros(N, bool).at[tg_j].set(same)
                ctl.update(rep_use=ru, rep_over=jnp.full((N,), -1.0))
            u_rec = jnp.zeros(K); eat_rec = jnp.zeros(K)
            if policy is not None:
                u = c["E"]; uf = jnp.where(sscale, fixv * c["E"][tg_j], fixv)
                u = u.at[tg_j].set(jnp.where(fr_T, u_t, jnp.where((fixc | sscale) & same, uf, u[tg_j])))
                ea = policy(u, c, pp); ea = ea.at[tg_j].set(jnp.where(cl_T, ea_t, ea[tg_j]))
                ctl.update(eat_use=sameN, eat_over=ea); u_rec = u[tg_j]; eat_rec = ea[tg_j]
            if haz is not None:
                ctl.update(haz=jnp.float32(haz), hu=jax.random.uniform(hk, (N,)))
            pre_t = c["t"]
            c2 = step(c, prm, ctl)
            own = c2["alive"][tg_j] & (c2["uid"][tg_j] == uid0[tg_j])
            ch = c2["alive"][None, :] & (c2["parent"][None, :] == uid0[tg_j][:, None]) & (c2["birth"][None, :] == pre_t)
            return c2, dict(E=jnp.where(own, c2["E"][tg_j], 0.0), alive=own, act=c2["act"][tg_j], x8=c2["xprev"][tg_j, 8],
                            u=u_rec, eat=eat_rec, births=ch.sum(1))
        _, rec = jax.lax.scan(body, s, (jax.random.split(hkey, T), aov, eov, xrec, urec, xval))
        return rec
    f = jax.jit(core)
    z8 = jnp.zeros((T, K, 8)); zk = jnp.zeros((T, K)); zb = jnp.zeros((T, K), bool)
    F, Tr = jnp.bool_(False), jnp.bool_(True)

    def run_arm(s, arm, ref=None, drop=0.5, key=None, hkey=None, fix=None, xrec=None, urec=None, pp=None):
        """arm in {'ref','cl','open','frozen','ablate','fixown','sscale'}; sscale = sensor-only intervention: sensed E <- fix * own E
        (fix = 1 - d), body untouched (drop forced 0); drop = fraction of E removed from targets at step 0 (ignored for ref).
        `xrec`/`urec` override the replayed sensed input (TEST USE ONLY: the deliberate-failure check feeds a reconstructed value)."""
        key = s["key"] if key is None else key; hkey = jax.random.PRNGKey(0) if hkey is None else hkey
        d = jnp.float32(0.0 if arm in ("ref", "fixown", "sscale") else drop)
        aov, eov, xr, ur, xv = z8, zk, zk, zk, zb
        if arm in ("open", "frozen"):
            assert ref is not None, "replay arms need the ref record"
            aov = jnp.asarray(ref["act"], jnp.float32)
            eov = jnp.asarray(ref["eat"]); xr = jnp.asarray(ref["x8"] if xrec is None else xrec, jnp.float32)
            ur = jnp.asarray(ref["u"] if urec is None else urec, jnp.float32)
            xv = jnp.asarray(np.concatenate([np.ones((1, K), bool), ref["alive"][:-1]], 0))
        fv = jnp.zeros(K) if fix is None else jnp.broadcast_to(jnp.asarray(fix, jnp.float32), (K,))
        rec = f(s, key, hkey, d, jnp.bool_(arm == "open"), jnp.bool_(arm == "frozen"), jnp.bool_(arm == "ablate"), jnp.bool_(arm == "fixown"), jnp.bool_(arm == "sscale"),
                aov, eov, xr, ur, xv, fv, jnp.zeros(1) if pp is None else jnp.asarray(pp, jnp.float32))
        return {k: np.asarray(v) for k, v in rec.items()}
    return run_arm

def identical(a, b, keys=("E", "alive", "act")):
    """bit-for-bit equality of two arm records (the no-change check)."""
    return all(np.array_equal(a[k], b[k]) for k in keys)

def post_step_lead_x8(ref):
    """The Exp 25/26 bug, expressed at sense level: the reference's POST-step energy turned into a channel-8 value and fed at the
    same step. Used ONLY by the deliberate-failure test (the identity check must reject it)."""
    return np.log1p(ref["E"]).astype(np.float32) - np.float32(0.7)
