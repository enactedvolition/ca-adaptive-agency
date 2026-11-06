"""Experiment 8: residence/speed threshold sweep + dynamic counterfactual replay on Exp 5 forks.

ARCHITECTURE.md §15 #3 and NIGHTLY_LOG 2026-11-05: after Exp 7's self-built = 0 under pe_cost+depletion
(and under sw_no_cost with the approximate own-deposit method), do (A) a threshold sweep and (B) a
true dynamic CF re-simulation. Forks store a scalar `own` (decayed local deposit EMA), not a lifetime
deposit map — CF_own subtracts that scalar from Wl at the agent's current cell (clamp ≥0). That is the
best available own-deposit proxy, not a recovered lifetime field. Dark-rooming ≠ agency; no agency claim.

  python exp8_counterfactual.py --procs 4
  python exp8_counterfactual.py --sweep-only
  python exp8_counterfactual.py --only sw_no_cost:0
"""
from __future__ import annotations
import argparse, os, sys, time, pickle, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("EXP8_OUT", os.path.join(HERE, "runs8"))
FORKS = os.path.join(HERE, "runs5")
VARIANTS = ["sw_full", "sw_pe", "sw_pecost", "sw_pecost_pe", "pecost_pe_static", "sw_pecost_pe_shuffled", "sw_no_cost"]
SEEDS = [0, 1, 2]
ENV_VAR = 0.35
N_LAYOUTS = 4
RES_GRID = [25, 50, 100, 200]
SPD_GRID = [0.1, 0.2, 0.4, 0.5]
DEFAULT_RES, DEFAULT_SPD = 100, 0.2
RELAX_RES, RELAX_SPD = 50, 0.4          # clearly-logged relaxed setting for pe_cost arms with ~0 dark
T = int(os.environ.get("EXP8_T", 100))
N_TGT = int(os.environ.get("EXP8_NTGT", 40))
BF = int(os.environ.get("EXP8_BF", 10))  # CF batch size (vmap)
DO_CF_CELL = os.environ.get("EXP8_CF_CELL", "1") != "0"


def layout_index(variant, t, params):
    per = float(params.get("sw_period", 0.0) or 0.0)
    if per <= 0:
        return 0
    return int(np.floor(float(t) / per)) % N_LAYOUTS


def amaps_from_kmaps(Kmaps):
    Kmaps = np.asarray(Kmaps, float)
    return 0.3 + Kmaps / Kmaps.max(axis=(1, 2), keepdims=True)


def classify_snap(snap, cfg, Amap, v_term, env_var, res_min, spd_frac, M2):
    reg = M2.regimes(snap, cfg, env_var, Amap, v_term, res_min=res_min, spd_frac=spd_frac)
    n = len(reg)
    frac = {name: float(np.mean(reg == i)) if n else float("nan") for i, name in enumerate(M2.REGIMES)}
    dark_any = float(sum(frac[k] for k in M2.REGIMES if k.startswith("dark"))) if n else float("nan")
    counts = {name: int((reg == i).sum()) for i, name in enumerate(M2.REGIMES)}
    return reg, frac, dark_any, counts


def build_snap(z, cfg, be):
    alive = np.asarray(z["alive"]).astype(bool)
    pos = np.asarray(z["pos"], float)
    cs = cfg.L / cfg.G
    cell = np.clip((pos / cs).astype(int), 0, cfg.G - 1)
    cx, cy = cell[:, 0], cell[:, 1]
    Wl = np.asarray(z["Wl"], float)
    a = alive
    snap = dict(
        uid=np.asarray(z["uid"])[a],
        res=np.asarray(z["res"], float)[a],
        spd=np.asarray(z["spd"], float)[a],
        own=np.asarray(z["own"], float)[a],
        cx=cx[a], cy=cy[a],
        W_here=Wl[cx, cy][a],
        build_eff=np.full(int(a.sum()), be, float),
    )
    # map alive-compact index -> global slot
    slots = np.nonzero(a)[0]
    return snap, slots, Wl, a


def sample_targets(reg_default, reg_relax, snap, slots, n_tgt, seed):
    """Prefer default-threshold dark agents; if few, add relax-dark then high-residence."""
    rng = np.random.default_rng(3000 + seed)
    dark_def = np.nonzero(np.isin(reg_default, [2, 3, 4]))[0]
    dark_rel = np.nonzero(np.isin(reg_relax, [2, 3, 4]))[0]
    order = []
    seen = set()
    for pool, tag in ((dark_def, "dark_default"), (dark_rel, "dark_relax")):
        pool = rng.permutation(pool)
        for i in pool:
            if i in seen:
                continue
            order.append((i, tag)); seen.add(i)
            if len(order) >= n_tgt:
                break
        if len(order) >= n_tgt:
            break
    if len(order) < n_tgt:
        # high residence among remaining alive
        rem = [i for i in range(len(snap["res"])) if i not in seen]
        rem = sorted(rem, key=lambda i: float(snap["res"][i]), reverse=True)
        for i in rem:
            order.append((i, "high_res")); seen.add(i)
            if len(order) >= n_tgt:
                break
    return order[:n_tgt]


def score_traj(rec, cfg, Amap, v_term, env_var, res_min, spd_frac, M2, be):
    """Score end-of-traj resident+quiet and leave vs start. rec keys: (K,T) or (T,) arrays."""
    # Support batched (K,T,...) and single (T,...)
    def _at(k):
        return np.asarray(rec[k])
    res = _at("res"); spd = _at("spd"); alive = _at("alive").astype(bool)
    own = _at("own"); W_here = _at("W_here"); surp = _at("surp")
    pestep = _at("pestep"); uid = _at("uid"); pos = _at("pos")
    # shapes: (K,T) or (T,)
    if res.ndim == 1:
        res = res[None]; spd = spd[None]; alive = alive[None]; own = own[None]
        W_here = W_here[None]; surp = surp[None]; pestep = pestep[None]
        uid = uid[None]; pos = pos[None]
    K, Tt = res.shape
    cx = np.clip((pos[..., 0] / (cfg.L / cfg.G)).astype(int), 0, cfg.G - 1)
    cy = np.clip((pos[..., 1] / (cfg.L / cfg.G)).astype(int), 0, cfg.G - 1)
    # end step
    end = Tt - 1
    # same-uid survival (vs uid at t=0 of this traj)
    uid0 = uid[:, 0]
    same = alive[:, end] & (uid[:, end] == uid0)
    # quiet at end via analytic OU
    amp = env_var * Amap[cx[:, end], cy[:, end]]
    gref = np.mean(M2.f_var(env_var * Amap, 0.0, cfg.theta, cfg.k_theta, cfg.k_wall))
    W = W_here[:, end] * be
    own_e = own[:, end] * be
    v_now = M2.f_var(amp, W, cfg.theta, cfg.k_theta, cfg.k_wall)
    quiet = v_now < 0.5 * gref
    resident = (res[:, end] >= res_min) & (spd[:, end] < spd_frac * v_term)
    darkish = same & resident & quiet
    # leave: died/replaced OR not (resident & quiet) at end
    leave = (~same) | ~(resident & quiet)
    # also: left starting cell
    left_cell = (cx[:, end] != cx[:, 0]) | (cy[:, end] != cy[:, 0])
    return dict(
        end_resident=resident.astype(bool),
        end_quiet=quiet.astype(bool),
        end_darkish=darkish.astype(bool),
        leave=leave.astype(bool),
        left_cell=left_cell.astype(bool),
        same_uid=same.astype(bool),
        end_res=res[:, end].astype(float),
        end_spd=spd[:, end].astype(float),
        end_surp=surp[:, end].astype(float),
        mean_pestep=pestep.mean(1).astype(float),
        mean_surp=surp.mean(1).astype(float),
        d_res=(res[:, end] - res[:, 0]).astype(float),
        d_spd=(spd[:, end] - spd[:, 0]).astype(float),
        d_surp=(surp[:, end] - surp[:, 0]).astype(float),
    )


def sweep_only_worker(job):
    variant, seed = job
    sys.path.insert(0, HERE)
    import sim5, metrics2 as M2, exp5_price as E5
    cfg = sim5.Cfg(N=2048, L=1.0, G=64)
    _, _, _, consts = sim5.make(cfg)
    Amaps = amaps_from_kmaps(consts["Kmaps"])
    prm = dict(E5.BASE, **E5.VARIANTS[variant])
    z = np.load(os.path.join(FORKS, f"fork_{variant}_s{seed}.npz"))
    t = int(np.asarray(z["t"])); li = layout_index(variant, t, prm); Amap = np.asarray(Amaps[li])
    be = float(prm.get("build_eff", 1.0))
    snap, slots, Wl, alive = build_snap(z, cfg, be)
    v_term = cfg.a_max / (1.0 - cfg.damp)
    grid = {}
    for rm in RES_GRID:
        for sf in SPD_GRID:
            reg, frac, dark_any, counts = classify_snap(snap, cfg, Amap, v_term, ENV_VAR, rm, sf, M2)
            key = f"res{rm}_spd{sf}"
            grid[key] = dict(res_min=rm, spd_frac=sf, regime_frac=frac, dark_any=dark_any,
                             regime_counts=counts, self_built=frac["dark: self-built"],
                             other_built=frac["dark: other-built"], natural=frac["dark: natural"])
    return dict(variant=variant, seed=seed, t=t, layout_index=li, build_eff=be, v_term=float(v_term),
                n_alive=int(alive.sum()), sweep=grid,
                note="Forks have scalar own (decayed local deposit EMA), not a lifetime deposit map.")


def worker(job):
    variant, seed = job
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    sys.path.insert(0, HERE)
    path = os.path.join(OUT, f"cf_{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0, "cached"
    import jax, jax.numpy as jnp, sim5, metrics2 as M2, exp5_price as E5
    t0 = time.time()
    cfg = sim5.Cfg(N=2048, L=1.0, G=64)
    init, chunk_one, colour, consts = sim5.make(cfg)
    step = consts["step"]; N = cfg.N; cs = cfg.L / cfg.G; G = cfg.G
    prm = E5.prm_of(variant)
    prm_dict = dict(E5.BASE, **E5.VARIANTS[variant])
    z = np.load(os.path.join(FORKS, f"fork_{variant}_s{seed}.npz"))
    tmpl = jax.eval_shape(init, jax.random.PRNGKey(0), prm)
    base = {k: (np.asarray(z[k]).astype(sd.dtype) if k in z.files else np.zeros(sd.shape, sd.dtype)) for k, sd in tmpl.items()}
    t = int(np.asarray(z["t"])); li = layout_index(variant, t, prm_dict)
    Amaps = amaps_from_kmaps(consts["Kmaps"]); Amap = np.asarray(Amaps[li])
    be = float(prm_dict.get("build_eff", 1.0))
    snap, slots_alive, Wl0, alive_mask = build_snap(z, cfg, be)
    v_term = cfg.a_max / (1.0 - cfg.damp)
    # ---- A) threshold sweep
    sweep = {}
    for rm in RES_GRID:
        for sf in SPD_GRID:
            reg, frac, dark_any, counts = classify_snap(snap, cfg, Amap, v_term, ENV_VAR, rm, sf, M2)
            key = f"res{rm}_spd{sf}"
            sweep[key] = dict(res_min=rm, spd_frac=sf, regime_frac=frac, dark_any=dark_any,
                              regime_counts=counts, self_built=float(frac["dark: self-built"]),
                              other_built=float(frac["dark: other-built"]), natural=float(frac["dark: natural"]))
    reg_def, _, _, _ = classify_snap(snap, cfg, Amap, v_term, ENV_VAR, DEFAULT_RES, DEFAULT_SPD, M2)
    reg_rel, _, _, _ = classify_snap(snap, cfg, Amap, v_term, ENV_VAR, RELAX_RES, RELAX_SPD, M2)
    targets = sample_targets(reg_def, reg_rel, snap, slots_alive, N_TGT, seed)
    # map compact -> global slot
    tg_slots = np.array([int(slots_alive[i]) for i, _ in targets], dtype=np.int32)
    tg_tags = [tag for _, tag in targets]
    tg_compact = np.array([i for i, _ in targets], dtype=np.int32)
    K = len(tg_slots)
    own0 = np.asarray(snap["own"][tg_compact], float)
    cx0 = np.asarray(snap["cx"][tg_compact], int)
    cy0 = np.asarray(snap["cy"][tg_compact], int)
    regime0 = np.asarray(reg_def[tg_compact], int)
    regime0_relax = np.asarray(reg_rel[tg_compact], int)
    # ---- B) dynamic CF replay
    base_j = {k: jnp.asarray(v) for k, v in base.items()}
    key0 = jnp.asarray(base["key"])

    def run_traj(s0, Wl_init):
        s = dict(s0); s["Wl"] = Wl_init
        def body(c, _):
            c = step(c, prm)
            cell = jnp.clip((c["pos"] / cs).astype(jnp.int32), 0, G - 1)
            W_here = c["Wl"][cell[:, 0], cell[:, 1]]
            rec = {k: c[k] for k in ("res", "spd", "surp", "pestep", "alive", "uid", "own", "pos", "E")}
            rec["W_here"] = W_here
            return c, rec
        _, recs = jax.lax.scan(body, s, None, length=T)
        return recs

    run_traj_jit = jax.jit(run_traj)

    # ref once (shared world)
    rec_ref_all = run_traj_jit(base_j, base_j["Wl"])
    # slice targets from full-N recordings -> (T, K, ...)
    def slice_tg(recs, slots):
        out = {}
        for k, v in recs.items():
            v = np.asarray(v)
            # v shape (T, N, ...) 
            out[k] = v[:, slots, ...]
        return out

    ref_tg = slice_tg(rec_ref_all, tg_slots)  # (T, K, ...)
    # transpose to (K, T, ...) for scoring
    def to_KT(d):
        return {k: (np.swapaxes(v, 0, 1) if v.ndim >= 2 else v) for k, v in d.items()}

    ref_KT = to_KT(ref_tg)

    # CF forks: modify Wl per target
    def make_Wl(mode, i):
        W = np.array(base["Wl"], dtype=np.float32, copy=True)
        if mode == "cf_own":
            W[cx0[i], cy0[i]] = max(0.0, float(W[cx0[i], cy0[i]] - own0[i]))
        elif mode == "cf_cell":
            W[cx0[i], cy0[i]] = 0.0
        return W

    def run_cf_batch(mode, indices):
        """Run CF for a list of target indices; return dict of (n,T,...) arrays."""
        # sequential jit calls in small batches (Wl differs per target; vmap over Wl)
        def run_one(Wl):
            return run_traj(base_j, Wl)
        run_b = jax.jit(jax.vmap(run_one))
        acc = []
        for i0 in range(0, len(indices), BF):
            chunk = indices[i0:i0 + BF]
            nreal = len(chunk)
            pads = list(chunk) + [chunk[-1]] * (BF - nreal)
            Wls = np.stack([make_Wl(mode, i) for i in pads]).astype(np.float32)
            rec = run_b(jnp.asarray(Wls))
            # rec[k]: (BF, T, N, ...)
            sliced = {}
            for k, v in rec.items():
                v = np.asarray(v)[:nreal]
                # pick each row's own target slot: (nreal, T, ...)
                slots_c = tg_slots[chunk]
                # v: (n, T, N, ...)
                gathered = []
                for j, sl in enumerate(slots_c):
                    gathered.append(v[j, :, sl, ...])
                sliced[k] = np.stack(gathered, 0)  # (n, T, ...)
            acc.append(sliced)
        return {k: np.concatenate([a[k] for a in acc], 0) for k in acc[0]}

    idxs = list(range(K))
    cf_own_KT = run_cf_batch("cf_own", idxs) if K else {}
    cf_cell_KT = run_cf_batch("cf_cell", idxs) if (DO_CF_CELL and K) else {}

    # Score under default and relaxed thresholds
    scores = {}
    for thr_name, rm, sf in (("default", DEFAULT_RES, DEFAULT_SPD), ("relax", RELAX_RES, RELAX_SPD)):
        sc_ref = score_traj(ref_KT, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be) if K else {}
        sc_own = score_traj(cf_own_KT, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be) if K else {}
        sc_cell = score_traj(cf_cell_KT, cfg, Amap, v_term, ENV_VAR, rm, sf, M2, be) if (DO_CF_CELL and K) else {}
        # paired deltas: CF − ref (positive leave Δ = CF made them leave more)
        def paired(a, b, key):
            if not a or not b: return {}
            return {
                f"d_{key}_leave_mean": float(np.mean(a["leave"].astype(float) - b["leave"].astype(float))),
                f"d_{key}_darkish_mean": float(np.mean(a["end_darkish"].astype(float) - b["end_darkish"].astype(float))),
                f"d_{key}_res_mean": float(np.mean(a["end_res"] - b["end_res"])),
                f"d_{key}_spd_mean": float(np.mean(a["end_spd"] - b["end_spd"])),
                f"d_{key}_surp_mean": float(np.mean(a["end_surp"] - b["end_surp"])),
                f"{key}_leave_rate": float(np.mean(a["leave"])),
                f"ref_leave_rate": float(np.mean(b["leave"])),
                f"{key}_darkish_rate": float(np.mean(a["end_darkish"])),
                f"ref_darkish_rate": float(np.mean(b["end_darkish"])),
            }
        block = dict(ref=sc_ref, cf_own=sc_own, cf_cell=sc_cell,
                     paired_own=paired(sc_own, sc_ref, "own"),
                     paired_cell=paired(sc_cell, sc_ref, "cell") if sc_cell else {})
        # behavioral self-built reclass: was dark (default) at fork, and CF_own leave > ref leave
        if K and sc_own and sc_ref:
            was_dark = np.isin(regime0, [2, 3, 4])
            was_other = regime0 == 3
            reclass = was_dark & sc_own["leave"] & (~sc_ref["leave"])
            reclass_other = was_other & sc_own["leave"] & (~sc_ref["leave"])
            block["n_was_dark"] = int(was_dark.sum())
            block["n_was_other"] = int(was_other.sum())
            block["n_reclass_self_beh"] = int(reclass.sum())
            block["n_reclass_other_to_self_beh"] = int(reclass_other.sum())
            block["frac_reclass_among_dark"] = float(reclass.sum() / max(was_dark.sum(), 1))
            block["frac_reclass_among_other"] = float(reclass_other.sum() / max(was_other.sum(), 1))
        scores[thr_name] = block

    out = dict(
        variant=variant, seed=seed, t=t, layout_index=li, build_eff=be, env_var=ENV_VAR,
        v_term=float(v_term), T=T, n_targets=K, n_alive=int(alive_mask.sum()),
        own_proxy_note=("Forks store scalar `own` = decayed local deposit EMA, not a lifetime deposit map. "
                        "cf_own subtracts own from Wl at the agent's fork cell (clamp ≥0). "
                        "This is a dynamic CF with the own proxy, not a recovered lifetime field."),
        sweep=sweep,
        targets=dict(
            slots=tg_slots, tags=tg_tags, own0=own0, cx0=cx0, cy0=cy0,
            regime0=regime0, regime0_relax=regime0_relax,
            regime_names_default=[M2.REGIMES[i] if 0 <= i < len(M2.REGIMES) else "undef" for i in regime0],
        ),
        scores=scores,
        wall_s=time.time() - t0,
        defaults=dict(res_min=DEFAULT_RES, spd_frac=DEFAULT_SPD, relax_res=RELAX_RES, relax_spd=RELAX_SPD),
    )
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    return path, out["wall_s"], "ok"


def smoke():
    sys.path.insert(0, HERE)
    import sim5, metrics2 as M2, exp5_price as E5
    variant, seed = "sw_no_cost", 0
    cfg = sim5.Cfg(N=2048, L=1.0, G=64)
    _, _, _, consts = sim5.make(cfg)
    Amaps = amaps_from_kmaps(consts["Kmaps"])
    prm = dict(E5.BASE, **E5.VARIANTS[variant])
    z = np.load(os.path.join(FORKS, f"fork_{variant}_s{seed}.npz"))
    t = int(np.asarray(z["t"])); li = layout_index(variant, t, prm); Amap = np.asarray(Amaps[li])
    be = float(prm.get("build_eff", 1.0))
    snap, slots, Wl, alive = build_snap(z, cfg, be)
    v_term = cfg.a_max / (1.0 - cfg.damp)
    print(f"smoke sweep {variant} s{seed} t={t} n_alive={alive.sum()} v_term={v_term:.6f}")
    for rm in (100, 50, 25):
        for sf in (0.2, 0.4):
            reg, frac, dark_any, counts = classify_snap(snap, cfg, Amap, v_term, ENV_VAR, rm, sf, M2)
            print(f"  res>={rm} spd<{sf}: self={frac['dark: self-built']:.4f} other={frac['dark: other-built']:.4f} "
                  f"nat={frac['dark: natural']:.4f} dark_any={dark_any:.4f} counts={counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--only", default=None, help="variant:seed")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        smoke(); sys.exit(0)
    if a.sweep_only:
        jobs = [(v, s) for v in VARIANTS for s in SEEDS]
        sweep_path = os.path.join(OUT, "sweep_all.pkl")
        results = [sweep_only_worker(j) for j in jobs]
        pickle.dump(results, open(sweep_path, "wb"))
        print(f"wrote {sweep_path} n={len(results)}")
        sys.exit(0)
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
