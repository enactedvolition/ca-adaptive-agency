"""Analysis + figures for sim2 experiments. Reads runs2/sweep and runs2/main pickles only."""
from __future__ import annotations
import glob, json, os, pickle, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exp2, metrics2 as M2, sim2
OUT = os.path.join(HERE, "out2"); os.makedirs(OUT, exist_ok=True)
BG = "#07080c"; FG = "#cdd"; DIM = "#789"
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
LAMCMAP = LinearSegmentedColormap.from_list("lam", [(0, "#2b6cff"), (0.5, "#ffffff"), (1, "#ff3b30")])
REG_COL = {"explore": "#7df9ff", "settled-noisy": "#888888", "dark: natural": "#3d4cff", "dark: other-built": "#c39bff", "dark: self-built": "#ffb000"}

def style(ax, title=None):
    ax.set_facecolor(BG); ax.tick_params(colors=DIM, labelsize=7)
    [s.set_color("#334") for s in ax.spines.values()]
    if title: ax.set_title(title, color=FG, fontsize=9, family="monospace", loc="left")

def ci(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float); n = len(x)
    if n < 2: return (np.nan, np.nan, np.nan, n)
    m = x.mean(); h = T975.get(n - 1, 2.0) * x.std(ddof=1) / np.sqrt(n)
    return (m, m - h, m + h, n)

def load(kind):
    R = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs2", kind, "batch_*.pkl"))):
        d = pickle.load(open(p, "rb")); R += d["results"]
    return R

def get(r, key):
    if key.startswith("lam_end."): return r["lam_end"][key.split(".")[1]]
    if key.startswith("lam0."): return r["lam0"][key.split(".")[1]]
    if key == "shift": return r["lam0"]["lam_absmed"] - r["lam_end"]["lam_absmed"]
    if key == "dark_any": return sum(r["regime_frac"][k] for k in M2.REGIMES if k.startswith("dark"))
    if key == "built": return r["regime_frac"]["dark: self-built"] + r["regime_frac"]["dark: other-built"]
    if key.startswith("reg:"): return r["regime_frac"][key[4:]]
    if key == "survived": return float(r["survived"])
    return r[key]

# ---------------------------------------------------------------- grid A
AG_METRICS = [("herit", "heritability of fitness (parent-offspring r)"), ("simpson_end", "lineage takeover (Simpson)"),
              ("A_col", "individuality proxy A=I(S';S|E) [nats]"), ("emp", "empowerment proxy I(A;X'|X) [nats]"),
              ("f_near", "fraction near-critical |λ|≤0.05"), ("lam_end.lam_med", "median λ_max (end)"),
              ("survived", "population survived (no rescue)")]

def grid_A(R):
    A = [r for r in R if r["job"]["grid"] == "A"]
    seeds = sorted({r["job"]["seed"] for r in A})
    def find(v, c=None, m=None, s=None):
        for r in A:
            j = r["job"]
            if j["variant"] == v and (c is None or j["c_basal"] == c) and (m is None or j["mut_sigma"] == m) and j["seed"] == s:
                return r
    tab = {}
    for key, _ in AG_METRICS + [("lam_end.f_near", "")]:
        k = "lam_end.f_near" if key == "f_near" else key
        full = np.full((len(exp2.COSTS), len(exp2.MUTS), len(seeds)), np.nan); dS = full.copy(); dM = full.copy(); dC = full.copy()
        for i, c in enumerate(exp2.COSTS):
            for jx, m in enumerate(exp2.MUTS):
                for si, s in enumerate(seeds):
                    f = find("full", c, m, s); sh = find("shuffled", c, m, s); nm = find("no_mut", c, None, s); nc = find("no_cost", None, m, s)
                    if f is None: continue
                    full[i, jx, si] = get(f, k)
                    if sh: dS[i, jx, si] = get(f, k) - get(sh, k)
                    if nm: dM[i, jx, si] = get(f, k) - get(nm, k)
                    if nc: dC[i, jx, si] = get(f, k) - get(nc, k)
        tab[key] = dict(full=full, d_shuffled=dS, d_no_mut=dM, d_no_cost=dC)
    return tab, seeds

def beats(dlist):
    """cell beats controls if every control-difference has 95% CI lower bound > 0 (paired by seed)."""
    out = np.zeros(dlist[0].shape[:2], bool); lows = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = all(ci(d[i, j])[1] > 0 for d in dlist)
    return out

def heat(ax, M, xt, yt, title, cmap="magma", center=None, mark=None, fmt="{:.2f}", xl="", yl=""):
    style(ax, title)
    if center is not None:
        vmax = np.nanmax(np.abs(M)) or 1
        im = ax.imshow(M, cmap="coolwarm", norm=TwoSlopeNorm(0, -vmax, vmax), origin="lower")
    else:
        im = ax.imshow(M, cmap=cmap, origin="lower")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5, color="#000" if center is not None else "#fff")
            if mark is not None and mark[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#7dff7d", lw=2))
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt); ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.set_xlabel(xl, color=DIM, fontsize=7); ax.set_ylabel(yl, color=DIM, fontsize=7)
    return im

def fig_phase_A(tab, seeds):
    keys = ["herit", "simpson_end", "A_col", "emp", "f_near", "lam_end.lam_med"]
    titles = dict(AG_METRICS)
    fig, axes = plt.subplots(3, len(keys), figsize=(4.0 * len(keys), 11), facecolor=BG)
    summary = {}
    for c, key in enumerate(keys):
        t = tab[key]
        mfull = np.nanmean(t["full"], 2)
        heat(axes[0, c], mfull, exp2.MUTS, exp2.COSTS, f"{titles[key]}\nfull, mean of n={len(seeds)}", xl="mutation σ", yl="existence cost")
        dS = np.nanmean(t["d_shuffled"], 2)
        b = beats([t["d_shuffled"], t["d_no_mut"], t["d_no_cost"]])
        heat(axes[1, c], dS, exp2.MUTS, exp2.COSTS, "full − shuffled heredity (paired)\n▢ green = beats ALL 3 controls, 95% CI", center=0, mark=b, xl="mutation σ", yl="existence cost")
        dmin = np.nanmin(np.stack([np.nanmean(t[k], 2) for k in ("d_shuffled", "d_no_mut", "d_no_cost")]), 0)
        heat(axes[2, c], dmin, exp2.MUTS, exp2.COSTS, "min over controls of (full − control)", center=0, mark=b, xl="mutation σ", yl="existence cost")
        summary[key] = dict(cells_beating_all=int(b.sum()), cells=b.size,
                            beating_cells=[(exp2.COSTS[i], exp2.MUTS[j]) for i, j in zip(*np.where(b))])
    fig.suptitle(f"Grid A: existence cost × mutation σ  (env_var={exp2.ENV_A}, N slots=512, 3000 steps, seeds n={len(seeds)}; t-CI df={len(seeds)-1})",
                 color="#eef", family="monospace", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "phase_A_agency_criticality.png"); fig.savefig(p, facecolor=BG, dpi=90); plt.close(fig)
    return p, summary

# ---------------------------------------------------------------- grid B
def grid_B(R):
    Bs = [r for r in R if r["job"]["grid"] == "B"]
    seeds = sorted({r["job"]["seed"] for r in Bs})
    keys = ["dark_any", "built", "reg:dark: self-built", "reg:dark: natural", "reg:explore", "reg:settled-noisy", "xvar_med", "surp_med", "pop_mean_late", "survived"]
    T = {v: {k: np.full((len(exp2.COSTS), len(exp2.ENVS), len(seeds)), np.nan) for k in keys} for v in ("full", "no_build", "shuffled")}
    for r in Bs:
        j = r["job"]; i = exp2.COSTS.index(j["c_basal"]); e = exp2.ENVS.index(j["env_var"]); s = seeds.index(j["seed"])
        for k in keys:
            T[j["variant"]][k][i, e, s] = get(r, k)
    return T, seeds, Bs

def fig_phase_B(T, seeds):
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), facecolor=BG)
    # composition map: RGB = (natural dark, built dark, explore)
    ax = axes[0, 0]; style(ax, f"behavioural regime map (full)\nR=found dark room · G=built dark room · B=explore")
    nat = np.nanmean(T["full"]["reg:dark: natural"], 2); bui = np.nanmean(T["full"]["built"], 2); exp_ = np.nanmean(T["full"]["reg:explore"], 2)
    tot = np.maximum(nat + bui + exp_, 1e-9)
    rgb = np.stack([nat / tot, bui / tot, exp_ / tot], -1)
    rgb = np.clip(rgb ** 0.5, 0, 1)
    ax.imshow(rgb, origin="lower")
    for i in range(nat.shape[0]):
        for j in range(nat.shape[1]):
            ax.text(j, i, f"N{nat[i,j]:.2f}\nB{bui[i,j]:.2f}\nE{exp_[i,j]:.2f}", ha="center", va="center", fontsize=6, color="#fff")
    ax.set_xticks(range(len(exp2.ENVS))); ax.set_xticklabels(exp2.ENVS); ax.set_yticks(range(len(exp2.COSTS))); ax.set_yticklabels(exp2.COSTS)
    ax.set_xlabel("environmental variability", color=DIM, fontsize=7); ax.set_ylabel("existence cost", color=DIM, fontsize=7)
    dB_sh = T["full"]["built"] - T["shuffled"]["built"]
    b_sh = beats([dB_sh])
    heat(axes[0, 1], np.nanmean(T["full"]["built"], 2), exp2.ENVS, exp2.COSTS, f"built dark rooms (self+other), full\n▢ = beats shuffled heredity, 95% CI n={len(seeds)}", mark=b_sh, xl="env variability", yl="existence cost")
    heat(axes[0, 2], np.nanmean(T["full"]["reg:dark: self-built"], 2), exp2.ENVS, exp2.COSTS, "self-built (counterfactually necessary) dark rooms", xl="env variability", yl="existence cost", fmt="{:.3f}")
    heat(axes[0, 3], np.nanmean(T["full"]["reg:dark: natural"], 2), exp2.ENVS, exp2.COSTS, "passive dark-rooming (naturally quiet spot)", xl="env variability", yl="existence cost")
    dx = T["full"]["xvar_med"] - T["no_build"]["xvar_med"]
    b_x = beats([-dx])
    heat(axes[1, 0], np.nanmean(dx, 2), exp2.ENVS, exp2.COSTS, "Δ median input variance, full − walls-inert\n(▢ = walls lower it, 95% CI)", center=0, mark=b_x, fmt="{:.3f}", xl="env variability", yl="existence cost")
    ds = T["full"]["surp_med"] - T["no_build"]["surp_med"]
    b_s = beats([-ds])
    heat(axes[1, 1], np.nanmean(ds, 2), exp2.ENVS, exp2.COSTS, "Δ median surprisal, full − walls-inert\n(▢ = walls lower it, 95% CI)", center=0, mark=b_s, xl="env variability", yl="existence cost")
    dp = T["full"]["pop_mean_late"] - T["no_build"]["pop_mean_late"]
    heat(axes[1, 2], np.nanmean(dp, 2), exp2.ENVS, exp2.COSTS, "Δ late population, full − walls-inert", center=0, fmt="{:.0f}", xl="env variability", yl="existence cost")
    heat(axes[1, 3], np.nanmean(T["full"]["survived"], 2), exp2.ENVS, exp2.COSTS, "survival fraction (full)", xl="env variability", yl="existence cost")
    fig.suptitle(f"Grid B: existence cost × environmental variability  (mutation σ={exp2.MUT_B}, seeds n={len(seeds)})", color="#eef", family="monospace", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "phase_B_darkroom.png"); fig.savefig(p, facecolor=BG, dpi=90); plt.close(fig)
    return p, dict(built_beats_shuffled=int(b_sh.sum()), walls_lower_xvar=int(b_x.sum()), walls_lower_surp=int(b_s.sum()),
                   built_cells=[(exp2.COSTS[i], exp2.ENVS[j]) for i, j in zip(*np.where(b_sh))])

def tradeoff(Bs):
    agg = {r: dict(n=0, surp=[], intake=[], rate=[], E=[], xvar=[]) for r in M2.REGIMES}
    for r in Bs:
        if r["job"]["variant"] != "full" or r["job"]["env_var"] == 0: continue
        for k, v in r["per_regime"].items():
            if v["n"] > 0:
                agg[k]["n"] += v["n"]
                for q in ("surp", "intake", "rate", "E", "xvar"):
                    agg[k][q].append((v[q], v["n"]))
    out = {}
    for k, a in agg.items():
        out[k] = dict(n=a["n"], **{q: (float(np.average([x for x, _ in a[q]], weights=[w for _, w in a[q]])) if a[q] else np.nan) for q in ("surp", "intake", "rate", "E", "xvar")})
    return out

# ---------------------------------------------------------------- main runs
def main_figs(Rm):
    by = {}
    for r in Rm: by.setdefault(r["job"]["variant"], []).append(r)
    variants = list(exp2.MAIN_VARIANTS)
    steps = Rm[0]["ts"]["step"]
    centers = 0.5 * (M2.LAM_BINS[1:] + M2.LAM_BINS[:-1])
    # ridgeline heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=BG, sharey=True)
    for ax, v in zip(axes.flat, variants):
        Hs = np.sum([r["ts"]["lam_hist"] for r in by[v]], 0).astype(float)
        Hn = Hs / np.maximum(Hs.sum(1, keepdims=True), 1)
        style(ax, f"{v}  (n={len(by[v])} seeds pooled)")
        ax.imshow(Hn.T, origin="lower", aspect="auto", extent=[steps[0], steps[-1], centers[0], centers[-1]], cmap="magma")
        for r in by[v]:
            ax.plot(steps, r["ts"]["lam_med"], color="#7df9ff", lw=0.8, alpha=0.8)
        ax.axhline(0, color="#fff", lw=0.6, ls="--"); ax.axhspan(-0.05, 0.05, color="#fff", alpha=0.08)
        ax.set_xlabel("step", color=DIM, fontsize=7); ax.set_ylabel("λ_max", color=DIM, fontsize=7)
    fig.suptitle("λ_max distribution over time (column-normalised); cyan = per-seed median; band = |λ|≤0.05", color="#eef", family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p_ridge_hm = os.path.join(OUT, "lambda_heatmap_main.png"); fig.savefig(p_ridge_hm, facecolor=BG, dpi=90); plt.close(fig)
    # classic ridgeline for full vs shuffled vs no_mut
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), facecolor=BG, sharey=True)
    for ax, v in zip(axes, ["full", "shuffled", "no_mut"]):
        style(ax, f"{v} — ridgeline (n={len(by[v])} seeds)")
        Hs = np.sum([r["ts"]["lam_hist"] for r in by[v]], 0).astype(float)
        idx = np.linspace(2, len(steps) - 1, 14).astype(int)
        for k, ti in enumerate(idx):
            h = Hs[ti] / max(Hs[ti].max(), 1)
            base = k * 0.6
            ax.fill_between(centers, base, base + h, color=plt.cm.cool(k / len(idx)), alpha=0.75, lw=0)
            ax.plot(centers, base + h, color="#fff", lw=0.4)
            ax.text(1.02, base + 0.1, f"{steps[ti]}", color=DIM, fontsize=6)
        ax.axvline(0, color="#fff", lw=0.6, ls="--"); ax.set_xlabel("λ_max", color=DIM, fontsize=8); ax.set_yticks([])
    fig.suptitle("λ_max ridgelines over time (bottom = early, top = late)", color="#eef", family="monospace")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p_ridge = os.path.join(OUT, "lambda_ridgeline.png"); fig.savefig(p_ridge, facecolor=BG, dpi=90); plt.close(fig)
    # λ vs fitness scatter (full, pooled seeds), plus spearman with bootstrap CI
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    stats = {}
    for ax, v in zip(axes, ["full", "shuffled"]):
        lam = np.concatenate([r["final"]["lam"] for r in by[v]]); rate = np.concatenate([r["final"]["rate"] for r in by[v]])
        age = np.concatenate([r["final"]["age"] for r in by[v]])
        ok = np.isfinite(lam) & (age >= 100)
        lam, rate = lam[ok], rate[ok]
        rs = []
        rng = np.random.default_rng(0)
        def spear(a, b):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
        rho = spear(lam, rate)
        for _ in range(500):
            ii = rng.integers(0, len(lam), len(lam)); rs.append(spear(lam[ii], rate[ii]))
        lo, hi = np.percentile(rs, [2.5, 97.5])
        # per-seed spearman
        per_seed = []
        for r in by[v]:
            l = r["final"]["lam"]; q = r["final"]["rate"]; a = r["final"]["age"]; o = np.isfinite(l) & (a >= 100)
            if o.sum() > 10: per_seed.append(spear(l[o], q[o]))
        stats[v] = dict(spearman=float(rho), boot95=(float(lo), float(hi)), n_agents=int(len(lam)), per_seed=[float(x) for x in per_seed])
        style(ax, f"{v}: λ_max vs reproduction rate (age≥100)\nSpearman {rho:.2f} [boot95 {lo:.2f},{hi:.2f}] n={len(lam)} agents / {len(by[v])} seeds")
        ax.scatter(lam, rate, s=3, c=lam, cmap=LAMCMAP, vmin=-0.5, vmax=0.5, linewidths=0)
        ax.axvline(0, color="#fff", lw=0.5, ls="--"); ax.set_xlabel("λ_max", color=DIM); ax.set_ylabel("offspring / step", color=DIM)
    fig.tight_layout(); p_sc = os.path.join(OUT, "lambda_vs_fitness.png"); fig.savefig(p_sc, facecolor=BG, dpi=90); plt.close(fig)
    # λ by lineage (full seed 0)
    r0 = by["full"][0]
    roots = np.concatenate([d["root"] for d in r0["dense"]])
    top = [x for x, _ in sorted(zip(*np.unique(roots, return_counts=True)), key=lambda z: -z[1])[:6]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=BG)
    style(axes[0], "median λ_max by lineage over time (full, seed 0, top 6 lineages)")
    style(axes[1], "mean surprisal by lineage over time (full, seed 0, agents age≥50)")
    for k, rt in enumerate(top):
        tt, lm, sm = [], [], []
        for d in r0["dense"]:
            m = d["root"] == rt
            if m.sum() >= 3:
                l = d["lam"][m].astype(float); l = l[np.isfinite(l)]
                s = d["surp"][m & (d["age"] >= 50)].astype(float)
                tt.append(d["t"]); lm.append(np.median(l) if len(l) else np.nan); sm.append(s.mean() if len(s) else np.nan)
        col = plt.cm.tab10(k)
        axes[0].plot(tt, lm, color=col, lw=1, label=f"root {rt}"); axes[1].plot(tt, sm, color=col, lw=1)
    axes[0].axhline(0, color="#fff", lw=0.5, ls="--"); axes[0].legend(fontsize=6, facecolor=BG, labelcolor=FG, edgecolor="#334")
    for a in axes: a.set_xlabel("step", color=DIM, fontsize=7)
    fig.tight_layout(); p_lin = os.path.join(OUT, "lineage_lambda_surprisal.png"); fig.savefig(p_lin, facecolor=BG, dpi=90); plt.close(fig)
    # field frames coloured by λ + variance overlay
    def field_frame(ax_l, ax_v, d, L=1.0, title=""):
        Fv = d["Fvar"].astype(float).T; Wl = d["Wl"].astype(float).T
        style(ax_l, f"agents coloured by λ_max (blue ordered · white critical · red chaotic) {title}")
        ax_l.imshow(np.sqrt(d["Rf"].astype(float).T), origin="lower", extent=[0, L, 0, L], cmap="Greys_r", vmin=0, vmax=4.0)
        lam = d["lam"].astype(float); ok = np.isfinite(lam)
        ax_l.scatter(d["pos"][~ok, 0], d["pos"][~ok, 1], s=2, c="#444", linewidths=0)
        ax_l.scatter(d["pos"][ok, 0], d["pos"][ok, 1], s=7, c=lam[ok], cmap=LAMCMAP, vmin=-0.5, vmax=0.5, linewidths=0)
        ax_l.set_xlim(0, L); ax_l.set_ylim(0, L); ax_l.set_xticks([]); ax_l.set_yticks([])
        style(ax_v, "local sensory variance (EMA of F²); cyan = wall density p90/p99")
        im = ax_v.imshow(Fv, origin="lower", extent=[0, L, 0, L], cmap="inferno")
        if Wl.max() > 0.05:
            lv = np.percentile(Wl[Wl > 0], [90, 99]) if (Wl > 0).sum() > 10 else [0.5]
            lv = np.unique(np.round(lv, 3))
            ax_v.contour(np.linspace(0, L, Wl.shape[1]), np.linspace(0, L, Wl.shape[0]), Wl, levels=lv, colors=["#7df9ff"], linewidths=0.7)
        ax_v.scatter(d["pos"][:, 0], d["pos"][:, 1], s=1, c="#ffffff", alpha=0.4, linewidths=0)
        ax_v.set_xlim(0, L); ax_v.set_ylim(0, L); ax_v.set_xticks([]); ax_v.set_yticks([])
        return im
    d_last = r0["dense"][-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor=BG)
    field_frame(axes[0], axes[1], d_last, title=f"step {d_last['t']}")
    fig.tight_layout(); p_field = os.path.join(OUT, "field_lambda_variance.png"); fig.savefig(p_field, facecolor=BG, dpi=90); plt.close(fig)
    # video
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), facecolor=BG, dpi=80)
    w, h = fig.canvas.get_width_height()
    mp4 = os.path.join(OUT, "lambda_field.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", "20", "-i", "-",
                           "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "22", mp4], stdin=subprocess.PIPE)
    for d in r0["dense"]:
        for a in axes: a.clear()
        field_frame(axes[0], axes[1], d, title=f"t={d['t']}")
        fig.canvas.draw(); ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    ff.stdin.close(); ff.wait(); plt.close(fig)
    # main-run metric table
    tab = {}
    for v in variants:
        tab[v] = {k: ci([get(r, k) for r in by[v]]) for k in ["lam_end.lam_med", "lam_end.lam_absmed", "lam_end.f_near", "lam0.lam_absmed", "shift", "herit", "simpson_end", "A_col", "A_star", "emp", "surp_med", "xvar_med", "dark_any", "built", "pop_mean_late", "rho_med", "gain_med"]}
    return dict(ridge_hm=p_ridge_hm, ridge=p_ridge, scatter=p_sc, lineage=p_lin, field=p_field, video=mp4), stats, tab, by

def dashboard(paths, out):
    import matplotlib.image as mpimg
    fig = plt.figure(figsize=(22, 15), facecolor=BG)
    spec = [("field", [0.0, 0.62, 0.40, 0.36]), ("ridge_hm", [0.40, 0.62, 0.36, 0.36]), ("scatter", [0.76, 0.70, 0.24, 0.20]),
            ("phaseA", [0.0, 0.25, 0.50, 0.36]), ("phaseB", [0.50, 0.25, 0.50, 0.36]), ("lineage", [0.0, 0.0, 0.50, 0.24]), ("tradeoff", [0.50, 0.0, 0.50, 0.24])]
    for k, rect in spec:
        if k not in paths: continue
        ax = fig.add_axes(rect); ax.imshow(mpimg.imread(paths[k])); ax.set_axis_off()
    fig.text(0.77, 0.66, "CA adaptive-agency v2 — dashboard", color="#eef", fontsize=13, family="monospace")
    fig.text(0.77, 0.645, "labels 'critical'/'agency' only where green-boxed (beats all controls, 95% CI)", color=DIM, fontsize=8, family="monospace")
    fig.savefig(out, facecolor=BG, dpi=80); plt.close(fig)

def fig_tradeoff(tr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=BG)
    regs = [r for r in M2.REGIMES if tr[r]["n"] > 0]
    for ax, (q, lab) in zip(axes, [("surp", "mean surprisal"), ("intake", "mean energy intake / step"), ("rate", "offspring / step")]):
        style(ax, f"{lab} by regime (grid B full runs, env>0)")
        ax.bar(range(len(regs)), [tr[r][q] for r in regs], color=[REG_COL[r] for r in regs])
        ax.set_xticks(range(len(regs))); ax.set_xticklabels([f"{r}\nn={tr[r]['n']}" for r in regs], fontsize=6, color=FG)
    fig.tight_layout(); p = os.path.join(OUT, "energy_surprise_tradeoff.png"); fig.savefig(p, facecolor=BG, dpi=90); plt.close(fig)
    return p

if __name__ == "__main__":
    paths, report = {}, {}
    Rs = load("sweep")
    if Rs:
        tabA, seedsA = grid_A(Rs)
        paths["phaseA"], report["gridA"] = fig_phase_A(tabA, seedsA)
        TB, seedsB, Bs = grid_B(Rs)
        paths["phaseB"], report["gridB"] = fig_phase_B(TB, seedsB)
        tr = tradeoff(Bs); report["tradeoff"] = tr; paths["tradeoff"] = fig_tradeoff(tr)
        # overall λ shift in grid A (full vs controls)
        sh = {v: [get(r, "shift") for r in Rs if r["job"]["grid"] == "A" and r["job"]["variant"] == v] for v in ("full", "shuffled", "no_mut", "no_cost")}
        lm = {v: [get(r, "lam_end.lam_med") for r in Rs if r["job"]["grid"] == "A" and r["job"]["variant"] == v] for v in sh}
        l0 = {v: [get(r, "lam0.lam_med") for r in Rs if r["job"]["grid"] == "A" and r["job"]["variant"] == v] for v in sh}
        fn = {v: [get(r, "lam_end.f_near") for r in Rs if r["job"]["grid"] == "A" and r["job"]["variant"] == v] for v in sh}
        report["gridA_lambda_pooled"] = {v: dict(n_runs=len(sh[v]), shift_abs_toward0=ci(sh[v]), lam_med_start=ci(l0[v]), lam_med_end=ci(lm[v]), f_near_end=ci(fn[v])) for v in sh}
        report["sweep_runs"] = len(Rs); report["sweep_overflow_total"] = int(sum(r["overflow"] for r in Rs))
    Rm = load("main")
    if Rm:
        mp, stats, tabm, by = main_figs(Rm); paths.update(mp); report["main_scatter"] = stats; report["main_table"] = tabm
        report["main_overflow_total"] = int(sum(r["overflow"] for r in Rm))
    dashboard(paths, os.path.join(OUT, "dashboard_v2.png")); paths["dashboard"] = os.path.join(OUT, "dashboard_v2.png")
    json.dump(dict(paths=paths, report=report), open(os.path.join(OUT, "report.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:12000])
