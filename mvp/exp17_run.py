"""Experiment 17: ungated fixed-cost reproduction (sim13 rep_fixed). Pre-registered in NIGHTLY_LOG 2026-11-19 before running.
  python exp17_run.py --bitcheck        sim13 rep_fixed=0 == sim12 on every field, bit for bit (sw_full s0-4, sw_pays s0-1, sw_pays_blind s0; 300 steps)
  python exp17_run.py --compete THR     mixed-competition gate at one threshold (forks: Exp 15b sw_full t=3000, seeds 0-2, as Exp 16), 400 steps
    Policy by lineage (root parity, heritable via genome donor): even = aware (rep = +1 iff true E > THR, else -1).
    Run A: odd = always (rep = +1).  Run B: odd = matched-rate blind (rep = +1 with prob p per step, p = aware lineages' mean rep
    indicator per live agent-step in run A of the same seed/THR).  World: rep_free = 1, rep_fixed = 1; brains drive all other actions.
    Outcome: aware share at t=0 vs mean over the last 50 steps (dshare). Writes runs17/compete_thr{THR}.json."""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs17"); sys.path.insert(0, HERE)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import exp16_run as E16
STEPS = 400; SEEDS = (0, 1, 2)

def bitcheck(steps=300):
    import sim12, sim13
    E16.VARIANTS = dict(E16.VARIANTS)
    out = {}
    for v, seeds in (("sw_full", (0, 1, 2, 3, 4)), ("sw_pays", (0, 1)), ("sw_pays_blind", (0,))):
        for sd in seeds:
            _, a, ca, pa = E16.make(sim12, v, [sd]); _, b, cb, pb = E16.make(sim13, v, [sd])
            for _ in range(steps // E16.DT):
                a = ca(a, pa, E16.DT); b = cb(b, pb, E16.DT)
            mm = [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True)]
            out[f"{v}_s{sd}"] = dict(n_fields_sim12=len(a), n_fields_sim13=len(b), mismatched=mm, births=float(np.asarray(b["c_rep"]).sum()),
                                     pop=int(np.asarray(b["alive"]).sum()))
            print(v, sd, out[f"{v}_s{sd}"], flush=True)
    _, b, cb, pb = E16.make(sim13, "sw_pays", [0], rep_fixed=1.0); _, a, ca, pa = E16.make(sim13, "sw_pays", [0])
    for _ in range(steps // E16.DT):
        a = ca(a, pa, E16.DT); b = cb(b, pb, E16.DT)
    out["mechanic_on_changes_dynamics"] = not np.array_equal(np.asarray(a["E"]), np.asarray(b["E"]))
    out["pop300_on_vs_off"] = [int(np.asarray(b["alive"]).sum()), int(np.asarray(a["alive"]).sum())]
    out["bit_for_bit"] = all(not r["mismatched"] for k, r in out.items() if isinstance(r, dict))
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "bitcheck.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}), flush=True)

def compete(thr, swap=0):
    par = 1 if swap else 0   # swap=1 (post hoc robustness): odd roots are aware, even roots are the comparator
    import jax, jax.numpy as jnp, sim13
    cfg = sim13.Cfg(N=2048, L=1.0, G=64, **E16.CFGKW); init, chunk_one, colour, consts = sim13.make(cfg); step = consts["step"]; N = cfg.N
    prm = sim13.params(**dict(E16.BASE, sw_period=E16.SW, rep_free=1.0, rep_fixed=1.0))
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    def state0(seed):
        z = np.load(os.path.join(E16.FORK15, f"fork15_sw_full_s{seed}_t3000.npz"))
        return {kk: (jnp.asarray(np.asarray(z[kk]).astype(sd.dtype)) if kk in z.files else jnp.zeros(sd.shape, sd.dtype)) for kk, sd in tmpl.items()}
    z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
    @jax.jit
    def simj(s, thr, mode, p, key):
        def body(c, k):
            aw = (c["root"] % 2) == par
            other = jnp.where(mode == 0, 1.0, jnp.where(jax.random.uniform(k, (N,)) < p, 1.0, -1.0))
            ro = jnp.where(aw, jnp.where(c["E"] > thr, 1.0, -1.0), other)
            c2 = step(c, prm, dict(aclamp=zb, aover=z8, e_off=zf, rep_use=jnp.ones(N, bool), rep_over=ro))
            al = c2["alive"]; a2 = al & ((c2["root"] % 2) == par); a0 = c["alive"]
            return c2, dict(pop=al.sum(), aw=a2.sum(), b_aw=(c2["repstep"] * aw).sum(), b_ot=(c2["repstep"] * ~aw).sum(),
                            att_aw=((ro > 0) & aw & a0).sum(), n_aw=(aw & a0).sum(), att_ot=((ro > 0) & ~aw & a0).sum(), n_ot=(~aw & a0).sum(),
                            E_aw=(c2["E"] * a2).sum() / jnp.maximum(a2.sum(), 1), E_ot=(c2["E"] * (al & ~a2)).sum() / jnp.maximum((al & ~a2).sum(), 1))
        _, r = jax.lax.scan(body, s, jax.random.split(key, STEPS))
        return r
    def one(sd, mode, p):
        s = state0(sd); al = np.asarray(s["alive"]); sh0 = float(((np.asarray(s["root"]) % 2 == par) & al).sum() / al.sum())
        r = {k: np.asarray(v) for k, v in simj(s, jnp.float32(thr), jnp.int32(mode), jnp.float32(p), jax.random.PRNGKey(171717 + sd)).items()}
        shT = float((r["aw"][-50:] / np.maximum(r["pop"][-50:], 1)).mean())
        return dict(seed=sd, share0=sh0, shareT=shT, dshare=shT - sh0, popT=float(r["pop"][-50:].mean()),
                    births_aware=float(r["b_aw"].sum()), births_other=float(r["b_ot"].sum()),
                    rate_aware=float(r["att_aw"].sum() / max(r["n_aw"].sum(), 1)), rate_other=float(r["att_ot"].sum() / max(r["n_ot"].sum(), 1)),
                    E_aware=float(r["E_aw"][-50:].mean()), E_other=float(r["E_ot"][-50:].mean()),
                    share_curve=(r["aw"] / np.maximum(r["pop"], 1))[::10].round(4).tolist())
    out = dict(thr=thr, swap=swap, steps=STEPS, seeds=list(SEEDS), E_endow=cfg.E_endow, E_over=cfg.E_over, vs_always=[], vs_blind=[])
    for sd in SEEDS:
        a = one(sd, 0, 0.0); out["vs_always"].append(a)
        b = one(sd, 1, a["rate_aware"]); b["p_matched"] = a["rate_aware"]; out["vs_blind"].append(b)
        print(thr, sd, json.dumps({k: v for k, v in a.items() if k != "share_curve"}), json.dumps({k: v for k, v in b.items() if k != "share_curve"}), flush=True)
    out["dshare_vs_always"] = [r["dshare"] for r in out["vs_always"]]; out["dshare_vs_blind"] = [r["dshare"] for r in out["vs_blind"]]
    os.makedirs(OUT, exist_ok=True); tmp = os.path.join(OUT, f"compete_thr{thr}{'_swap' if swap else ''}.tmp.json")
    json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(OUT, f"compete_thr{thr}{'_swap' if swap else ''}.json"))
    print("DONE", thr, out["dshare_vs_always"], out["dshare_vs_blind"], flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bitcheck", action="store_true"); ap.add_argument("--compete", type=float, default=None); ap.add_argument("--swap", type=int, default=0)
    a = ap.parse_args()
    if a.bitcheck: bitcheck()
    if a.compete is not None: compete(a.compete, a.swap)
