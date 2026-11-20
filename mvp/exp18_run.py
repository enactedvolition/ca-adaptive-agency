"""Experiment 18: re-gate sim13 with the blind comparator at its own best rate. Pre-registered in NIGHTLY_LOG 2026-11-20 before any data.
  python exp18_run.py --sweep --seeds 0 [--ecap 4.0]     blind(p) (even roots) vs always (odd roots), p over GRID; one JSON per job in runs18/
  python exp18_run.py --gate  --seeds 0,1,2 [--ecap 4.0] [--swap 0|1]   aware(THR 0.8) vs blind(p*), p* = argmax mean sweep dshare over seeds 0-2
Same world/forks/protocol as exp17_run.compete (sim13, rep_free=1, rep_fixed=1, Exp 15b sw_full t=3000 forks, 400 steps, last-50 share).
Resumable: finished jobs (JSON present) are skipped."""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs18"); sys.path.insert(0, HERE)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import exp16_run as E16
STEPS = 400; GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05); THR = 0.8

def make_runner(ecap):
    import jax, jax.numpy as jnp, sim13
    kw = dict(E16.CFGKW); cfg = sim13.Cfg(N=2048, L=1.0, G=64, E_cap=float(ecap), **kw); init, chunk_one, colour, consts = sim13.make(cfg); step = consts["step"]; N = cfg.N
    prm = sim13.params(**dict(E16.BASE, sw_period=E16.SW, rep_free=1.0, rep_fixed=1.0))
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    def state0(seed):
        z = np.load(os.path.join(E16.FORK15, f"fork15_sw_full_s{seed}_t3000.npz"))
        return {kk: (jnp.asarray(np.asarray(z[kk]).astype(sd.dtype)) if kk in z.files else jnp.zeros(sd.shape, sd.dtype)) for kk, sd in tmpl.items()}
    z8 = jnp.zeros((N, 8)); zb = jnp.zeros(N, bool); zf = jnp.zeros(N)
    @jax.jit
    def simj(s, par, modeA, thr, pA, pB, key):
        # slot A = roots with parity par: modeA 0 = aware(thr), 1 = blind(pA).  slot B = other parity: blind(pB) (pB >= 1 => always).
        def body(c, k):
            kA, kB = jax.random.split(k)
            aw = (c["root"] % 2) == par
            ra = jnp.where(modeA == 0, jnp.where(c["E"] > thr, 1.0, -1.0), jnp.where(jax.random.uniform(kA, (N,)) < pA, 1.0, -1.0))
            rb = jnp.where(jax.random.uniform(kB, (N,)) < pB, 1.0, -1.0)
            ro = jnp.where(aw, ra, rb)
            c2 = step(c, prm, dict(aclamp=zb, aover=z8, e_off=zf, rep_use=jnp.ones(N, bool), rep_over=ro))
            al = c2["alive"]; a2 = al & ((c2["root"] % 2) == par); a0 = c["alive"]
            return c2, dict(pop=al.sum(), aw=a2.sum(), b_aw=(c2["repstep"] * aw).sum(), b_ot=(c2["repstep"] * ~aw).sum(),
                            att_aw=((ra > 0) & aw & a0).sum(), n_aw=(aw & a0).sum(), att_ot=((rb > 0) & ~aw & a0).sum(), n_ot=(~aw & a0).sum(),
                            E_aw=(c2["E"] * a2).sum() / jnp.maximum(a2.sum(), 1), E_ot=(c2["E"] * (al & ~a2)).sum() / jnp.maximum((al & ~a2).sum(), 1))
        _, r = jax.lax.scan(body, s, jax.random.split(key, STEPS)); return r
    def one(sd, par, modeA, thr, pA, pB):
        s = state0(sd); al = np.asarray(s["alive"]); sh0 = float(((np.asarray(s["root"]) % 2 == par) & al).sum() / al.sum())
        r = {k: np.asarray(v) for k, v in simj(s, jnp.int32(par), jnp.int32(modeA), jnp.float32(thr), jnp.float32(pA), jnp.float32(pB), jax.random.PRNGKey(181818 + sd)).items()}
        shT = float((r["aw"][-50:] / np.maximum(r["pop"][-50:], 1)).mean())
        return dict(seed=sd, share0=sh0, shareT=shT, dshare=shT - sh0, popT=float(r["pop"][-50:].mean()), births_A=float(r["b_aw"].sum()), births_B=float(r["b_ot"].sum()),
                    rate_A=float(r["att_aw"].sum() / max(r["n_aw"].sum(), 1)), rate_B=float(r["att_ot"].sum() / max(r["n_ot"].sum(), 1)),
                    E_A=float(r["E_aw"][-50:].mean()), E_B=float(r["E_ot"][-50:].mean()), share_curve=(r["aw"] / np.maximum(r["pop"], 1))[::10].round(4).tolist())
    return one

def pstar(ecap, seeds=(0, 1, 2)):
    m = {}
    for p in GRID:
        v = [json.load(open(os.path.join(OUT, f"sweep_ecap{ecap}_p{p}_s{sd}.json")))["dshare"] for sd in seeds]
        m[p] = float(np.mean(v))
    return max(m, key=m.get), m

def save(name, d):
    os.makedirs(OUT, exist_ok=True); t = os.path.join(OUT, name + ".tmp"); json.dump(d, open(t, "w"), indent=1); os.replace(t, os.path.join(OUT, name + ".json"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sweep", action="store_true"); ap.add_argument("--gate", action="store_true")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--ecap", type=float, default=4.0); ap.add_argument("--swap", type=int, default=0); ap.add_argument("--plist", default=None); ap.add_argument("--gatep", type=float, default=None)
    a = ap.parse_args(); seeds = [int(x) for x in a.seeds.split(",")]; one = make_runner(a.ecap)
    if a.sweep:
        for sd in seeds:
            for p in (GRID if a.plist is None else [float(x) for x in a.plist.split(",")]):
                nm = f"sweep_ecap{a.ecap}_p{p}_s{sd}"
                if os.path.exists(os.path.join(OUT, nm + ".json")): continue
                r = one(sd, 0, 1, THR, p, 1.0); r.update(p=p, ecap=a.ecap); save(nm, r)
                print(nm, json.dumps({k: v for k, v in r.items() if k != "share_curve"}), flush=True)
    if a.gate:
        ps, m = pstar(a.ecap); print("p*", ps, m, flush=True)
        tag = ""
        if a.gatep is not None: ps = a.gatep; tag = f"_p{ps}"   # post hoc sensitivity: gate against a named grid p instead of p*
        for sd in seeds:
            nm = f"gate_ecap{a.ecap}_swap{a.swap}{tag}_s{sd}"
            if os.path.exists(os.path.join(OUT, nm + ".json")): continue
            r = one(sd, a.swap, 0, THR, 0.0, ps); r.update(pstar=ps, ecap=a.ecap, swap=a.swap, sweep_means=m); save(nm, r)
            print(nm, json.dumps({k: v for k, v in r.items() if k not in ("share_curve", "sweep_means")}), flush=True)
