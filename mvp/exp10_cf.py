"""Experiment 10b: exact self-built classification + true counterfactual replay on sim8 (guard) forks at step 2500.
Same protocol as exp9_cf.py, but replay runs sim8 with attribution on, because in guard arms ownership (Adep) feeds the dynamics:
  ref     : unchanged
  cf_true : Wl - Adep[agent] (clamp >= 0) AND Adep[agent] = 0  (its walls and its claim on them are gone)
  cf_ctrl : remove the same mass from walls >= CTRL_R cells away, proportional to Wl, scaling Adep/Aorph there by the same
            factor (ownership shares elsewhere preserved; equal-mass, elsewhere)
Extra outcome in guard arms: change in the agent's intake EMA and energy at the end of the replay (does losing its own walls cost it?).
Dark-rooming is not agency; no agency claim.
  python exp10_cf.py --procs 6
"""
from __future__ import annotations
import argparse, os, sys, time, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP10_OUT", os.path.join(HERE, "runs10"))
VARIANTS = ["sw_full", "sw_guard", "sw_guard_inert", "sw_guard_shuffled", "sw_pecost_pe", "sw_pecost_pe_guard"]
SEEDS = [0, 1, 2]
ENV_VAR = 0.35
RES_GRID = [25, 50, 100, 200]; SPD_GRID = [0.1, 0.2, 0.4, 0.5]
CORNERS = [(100, 0.2), (25, 0.1), (25, 0.5), (200, 0.1), (200, 0.5)]
DEFAULT_RES, DEFAULT_SPD = 100, 0.2
RELAX_RES, RELAX_SPD = 50, 0.4
T = int(os.environ.get("EXP10_T", 100)); N_TGT = int(os.environ.get("EXP10_NTGT", 40)); BF = int(os.environ.get("EXP10_BF", 10))
CTRL_R = 4

def worker(job):
    variant, seed = job
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)
    path = os.path.join(OUT, f"cf10_{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0, "cached"
    import jax, jax.numpy as jnp, sim8, metrics2 as M2, exp10_run as E10, exp8_counterfactual as E8
    t0 = time.time()
    cfg = sim8.Cfg(N=2048, L=1.0, G=64)
    init, chunk_one, colour, consts = sim8.make(cfg)
    step = consts["step"]; N = cfg.N; cs = cfg.L / cfg.G; G = cfg.G
    prm_dict = dict(E10.BASE, **E10.VARIANTS[variant]); prm = sim8.params(**prm_dict)
    z = np.load(os.path.join(OUT, f"fork10_{variant}_s{seed}.npz"))
    Adep = z["Adep"]
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    base = {k: np.asarray(z[k]).astype(sd.dtype) for k, sd in tmpl.items()}
    t = int(base["t"]); li = E8.layout_index(variant, t, prm_dict)
    Amap = np.asarray(E8.amaps_from_kmaps(consts["Kmaps"])[li])
    be = float(prm_dict.get("build_eff", 1.0))
    snap, slots_alive, Wl0, alive_mask = E8.build_snap(z, cfg, be)
    v_term = cfg.a_max / (1.0 - cfg.damp)
    own_exact = Adep[slots_alive, snap["cx"], snap["cy"]].astype(float)
    snap_ex = dict(snap, own=own_exact)
    # ---- (i) static classification, proxy vs exact
    static = {}
    for rm in RES_GRID:
        for sf in SPD_GRID:
            row = {}
            for name, sn in (("proxy", snap), ("exact", snap_ex)):
                reg, frac, dark_any, counts = E8.classify_snap(sn, cfg, Amap, v_term, ENV_VAR, rm, sf, M2)
                row[name] = dict(regime_frac=frac, regime_counts=counts, dark_any=dark_any, self_built=frac["dark: self-built"],
                                 other_built=frac["dark: other-built"], natural=frac["dark: natural"])
            static[f"res{rm}_spd{sf}"] = row
    reg_px, _, _, _ = E8.classify_snap(snap, cfg, Amap, v_term, ENV_VAR, DEFAULT_RES, DEFAULT_SPD, M2)
    reg_ex, _, _, _ = E8.classify_snap(snap_ex, cfg, Amap, v_term, ENV_VAR, DEFAULT_RES, DEFAULT_SPD, M2)
    reg_exr, _, _, _ = E8.classify_snap(snap_ex, cfg, Amap, v_term, ENV_VAR, RELAX_RES, RELAX_SPD, M2)
    dark = np.isin(reg_ex, [2, 3, 4])
    Wh = np.maximum(snap["W_here"], 1e-12)
    own_share = dict(
        dark_mean_exact_share=float(np.mean(own_exact[dark] / Wh[dark])) if dark.any() else float("nan"),
        dark_max_exact_share=float(np.max(own_exact[dark] / Wh[dark])) if dark.any() else float("nan"),
        dark_mean_proxy_share=float(np.mean(snap["own"][dark] / Wh[dark])) if dark.any() else float("nan"),
        all_mean_exact_share=float(np.mean(own_exact / Wh)), all_mean_proxy_share=float(np.mean(snap["own"] / Wh)),
        exact_vs_proxy_corr=float(np.corrcoef(own_exact, snap["own"])[0, 1]) if len(own_exact) > 2 else float("nan"),
        n_dark=int(dark.sum()), n_alive=int(len(own_exact)),
        n_disagree_default=int((reg_px != reg_ex).sum()),
        lifetime_mass_alive_mean=float(Adep[slots_alive].sum((1, 2)).mean()),
        frac_wall_mass_by_living=float(Adep.sum() / max(Wl0.sum(), 1e-12)),
    )
    # ---- (ii) dynamic CF
    targets = E8.sample_targets(reg_ex, reg_exr, snap, slots_alive, N_TGT, seed)
    tg_c = np.array([i for i, _ in targets], dtype=np.int64); tg_tags = [tag for _, tag in targets]
    tg_slots = slots_alive[tg_c].astype(np.int64); K = len(tg_slots)
    cx0 = snap["cx"][tg_c]; cy0 = snap["cy"][tg_c]
    ii = np.arange(G)
    def make_Wl(mode, k):
        """Returns ((Wl, Adep_row_scale_map or None, zero_slot or -1), removed mass). Adep edits are applied on device."""
        W = np.array(base["Wl"], dtype=np.float64, copy=True)
        A = Adep[tg_slots[k]].astype(np.float64)
        if mode == "cf_true":
            W = np.maximum(W - A, 0.0)
            return (W.astype(np.float32), np.ones((G, G), np.float32), int(tg_slots[k])), float(np.minimum(A, np.array(base["Wl"], dtype=np.float64)).sum())
        m = float(np.minimum(A, W).sum())           # mass actually removable by cf_true (clamp), matched by ctrl
        dx = np.abs(ii - cx0[k]); dx = np.minimum(dx, G - dx); dy = np.abs(ii - cy0[k]); dy = np.minimum(dy, G - dy)
        far = (np.maximum(dx[:, None], dy[None, :]) >= CTRL_R)
        avail = W * far; tot = avail.sum()
        frac = min(1.0, m / tot) if tot > 0 else 0.0
        W = W - avail * frac
        scale = np.where(far, 1.0 - frac, 1.0).astype(np.float32)   # ownership scaled with the wall it describes
        return (W.astype(np.float32), scale, -1), float(avail.sum() * frac)
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    def run_traj(Wl_init, scale=None, zslot=None):
        s = dict(base_j); s["Wl"] = Wl_init
        if scale is not None:
            keep = (jnp.arange(N) != zslot).astype(jnp.float32)
            s["Adep"] = s["Adep"] * scale[None] * keep[:, None, None]; s["Aorph"] = s["Aorph"] * scale
        def body(c, _):
            c = step(c, prm)
            cell = jnp.clip((c["pos"] / cs).astype(jnp.int32), 0, G - 1)
            rec = {k: c[k] for k in ("res", "spd", "surp", "pestep", "alive", "uid", "own", "pos", "E", "intake", "clstep")}
            rec["W_here"] = c["Wl"][cell[:, 0], cell[:, 1]]
            return c, rec
        _, recs = jax.lax.scan(body, s, None, length=T)
        return recs
    run1 = jax.jit(run_traj)
    rec_ref = {k: np.swapaxes(np.asarray(v)[:, tg_slots], 0, 1) for k, v in run1(base_j["Wl"]).items()}
    run_b = jax.jit(jax.vmap(run_traj))
    assert np.array_equal(np.asarray(jax.jit(run_traj)(base_j["Wl"], jnp.ones((G, G)), jnp.int32(-1))["E"]), np.asarray(run1(base_j["Wl"])["E"]))   # identity edit == ref
    masses = {}
    def run_mode(mode):
        acc = []; ms = []
        for i0 in range(0, K, BF):
            ch = list(range(i0, min(K, i0 + BF))); nreal = len(ch); pads = ch + [ch[-1]] * (BF - nreal)
            Wm = [make_Wl(mode, k) for k in pads]; ms += [w[1] for w in Wm[:nreal]]
            rec = run_b(jnp.asarray(np.stack([w[0][0] for w in Wm])), jnp.asarray(np.stack([w[0][1] for w in Wm])), jnp.asarray(np.array([w[0][2] for w in Wm], np.int32)))
            acc.append({k: np.stack([np.asarray(v)[j, :, tg_slots[c]] for j, c in enumerate(ch)]) for k, v in rec.items()})
        masses[mode] = np.array(ms)
        return {k: np.concatenate([a[k] for a in acc], 0) for k in acc[0]}
    rec_true = run_mode("cf_true") if K else {}
    rec_ctrl = run_mode("cf_ctrl") if K else {}
    scores = {}
    regime0 = reg_ex[tg_c]; regime0_px = reg_px[tg_c]
    for thr, rm, sf in (("default", DEFAULT_RES, DEFAULT_SPD), ("relax", RELAX_RES, RELAX_SPD)):
        if not K: scores[thr] = {}; continue
        sr = E8.score_traj(rec_ref, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be)
        stt = E8.score_traj(rec_true, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be)
        sc = E8.score_traj(rec_ctrl, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be)
        L = lambda d: d["leave"].astype(float)
        was_dark = np.isin(regime0, [2, 3, 4]); was_other = regime0 == 3
        beh = was_dark & stt["leave"] & ~sr["leave"] & ~sc["leave"]
        beh_o = was_other & stt["leave"] & ~sr["leave"] & ~sc["leave"]
        endv = lambda r, k: np.where(r["alive"][:, -1] & (r["uid"][:, -1] == r["uid"][:, 0]), r[k][:, -1], np.nan)
        scores[thr] = dict(ref=sr, cf_true=stt, cf_ctrl=sc,
                           d_intake_true_ctrl=float(np.nanmean(endv(rec_true, "intake") - endv(rec_ctrl, "intake"))),
                           d_E_true_ctrl=float(np.nanmean(endv(rec_true, "E") - endv(rec_ctrl, "E"))),
                           d_claim_true_ctrl=float(np.nanmean(rec_true["clstep"][:, -1] - rec_ctrl["clstep"][:, -1])),
                           d_true=float(np.mean(L(stt) - L(sr))), d_ctrl=float(np.mean(L(sc) - L(sr))), d_true_ctrl=float(np.mean(L(stt) - L(sc))),
                           d_true_ctrl_dark=float(np.mean((L(stt) - L(sc))[was_dark])) if was_dark.any() else float("nan"),
                           ref_leave=float(L(sr).mean()), true_leave=float(L(stt).mean()), ctrl_leave=float(L(sc).mean()),
                           n_was_dark=int(was_dark.sum()), n_was_other=int(was_other.sum()), n_beh_self=int(beh.sum()), n_beh_other_to_self=int(beh_o.sum()),
                           frac_beh_self_among_dark=float(beh.sum() / max(was_dark.sum(), 1)),
                           frac_beh_other_to_self=float(beh_o.sum() / max(was_other.sum(), 1)))
    out = dict(variant=variant, seed=seed, t=t, layout_index=li, T=T, n_targets=K, n_alive=int(alive_mask.sum()), ctrl_r=CTRL_R,
               static=static, own_share=own_share,
               targets=dict(slots=tg_slots, tags=tg_tags, regime0_exact=regime0, regime0_proxy=regime0_px, cx0=cx0, cy0=cy0,
                            own_exact=own_exact[tg_c], own_proxy=snap["own"][tg_c],
                            mass_true=masses.get("cf_true"), mass_ctrl=masses.get("cf_ctrl")),
               scores=scores, wall_s=time.time() - t0)
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, out["wall_s"], "ok"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--only", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    if a.only:
        v, s = a.only.split(":"); jobs = [(v, int(s))]
    t0 = time.time()
    if a.procs <= 1 or len(jobs) == 1:
        for j in jobs:
            p, w, d = worker(j); print("DONE", os.path.basename(p), f"{w:.1f}s", d, flush=True)
    else:
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for p, w, d in pool.imap_unordered(worker, jobs):
                print("DONE", os.path.basename(p), f"{w:.1f}s", d, f"elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"all done in {time.time()-t0:.1f}s", flush=True)
