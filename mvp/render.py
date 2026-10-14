"""Render the four control runs side by side (honesty layer) + an analysis
dashboard. Reads only the zarr snapshots + provenance written by sim.py:
the render layer never touches simulation state (sim/render decoupling)."""
from __future__ import annotations
import json, os, subprocess, sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(__file__), "runs")
OUT = os.path.join(os.path.dirname(__file__), "out")
VARIANTS = ["full", "no_cost", "shuffled", "memoryless"]
TITLES = {"full": "full model", "no_cost": "control: energy is free",
          "shuffled": "control: shuffled heredity", "memoryless": "control: no recurrence (h=0)"}
VCOL = {"full": "#7df9ff", "no_cost": "#ffb347", "shuffled": "#ff6fae", "memoryless": "#a6ff7d"}
PX = 384
LOW_FIDELITY = 0.5          # same threshold Mottled uses (projection.LOW_FIDELITY)
BG = "#07080c"

def load(v):
    g = zarr.open_group(os.path.join(RUNS, v, "snapshots.zarr"), mode="r")
    prov = json.load(open(os.path.join(RUNS, v, "provenance.json")))
    return g, prov

def clusters(pos, thresh):
    """Connected components of agents closer than `thresh` (torus)."""
    n = len(pos)
    if n == 0:
        return np.zeros(0, int)
    d = pos[None] - pos[:, None]; d -= np.round(d)
    adj = (d ** 2).sum(-1) < thresh ** 2
    lab = np.arange(n)
    while True:
        new = np.where(adj, lab[None, :], n).min(1)
        new = np.minimum(new, lab)
        new = new[new]                       # pointer jumping
        if np.array_equal(new, lab):
            return lab
        lab = new

def metrics(g, prov):
    cfg = prov["config"]; T = g["alive"].shape[0]
    thr = 1.9 * cfg["r0"]
    out = {k: np.zeros(T) for k in ("pop", "E", "lineages", "clusters", "in_clusters", "kin_purity")}
    budget = np.zeros((T, 6))
    for t in range(T):
        m = g["alive"][t]; pos = g["pos"][t][m]; root = g["root"][t][m]; a = g["act"][t][m].astype(float)
        out["pop"][t] = m.sum(); out["E"][t] = g["E"][t][m].astype(float).mean() if m.any() else 0
        out["lineages"][t] = len(np.unique(root))
        lab = clusters(pos, thr)
        ids, cnt = np.unique(lab, return_counts=True)
        big = ids[cnt >= 4]
        out["clusters"][t] = len(big)
        out["in_clusters"][t] = np.isin(lab, big).mean() if len(lab) else 0
        pur = [np.unique(root[lab == b], return_counts=True)[1].max() / (lab == b).sum() for b in big]
        out["kin_purity"][t] = np.mean(pur) if pur else np.nan
        if m.any():
            budget[t] = [cfg["c_basal"], cfg["c_move"] * (a[:, :2] ** 2).sum(1).mean(),
                         cfg["c_eat"] * ((a[:, 2] + 1) / 2).mean(), cfg["c_sig"] * np.abs(a[:, 5]).mean(),
                         cfg["c_bond"] * np.maximum(a[:, 6], 0).mean(), cfg["c_att"] * np.maximum(-a[:, 4], 0).mean()]
    return out, budget

def field_rgb(f):
    f = np.clip(f.astype(float) / 1.2, 0, 1)
    rgb = np.stack([0.10 * f, 0.22 * f, 0.30 * f], -1) + 0.012
    k = PX // f.shape[0]
    return np.kron(rgb.transpose(1, 0, 2), np.ones((k, k, 1)))   # field[x, y] -> image[row=y, col=x]

def splat(img, pos, col, E, r=1):
    xy = np.clip((pos * PX).astype(int), 0, PX - 1)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + 1:
                continue
            x = (xy[:, 0] + dx) % PX; y = (xy[:, 1] + dy) % PX
            img[y, x] = np.maximum(img[y, x], col * np.clip(0.55 + 0.25 * E[:, None], 0.4, 1.2))
    return img

def video(gs, ms, frames_png_at):
    T = min(g["alive"].shape[0] for g, _ in gs.values())
    fig = plt.figure(figsize=(10, 11.4), dpi=96, facecolor=BG)
    axes = [fig.add_axes([0.01 + 0.5 * (i % 2), 0.32 + 0.34 * (1 - i // 2), 0.48, 0.32]) for i in range(4)]
    axt = fig.add_axes([0.08, 0.05, 0.88, 0.22], facecolor=BG)
    trails = {v: np.zeros((PX, PX, 3)) for v in VARIANTS}
    ims = {}
    for ax, v in zip(axes, VARIANTS):
        ax.set_axis_off()
        ims[v] = ax.imshow(np.zeros((PX, PX, 3)), interpolation="nearest")
        ax.set_title(TITLES[v], color=VCOL[v], fontsize=11, family="monospace")
    for v in VARIANTS:
        axt.plot(np.arange(T) * 10, ms[v][0]["pop"][:T], color=VCOL[v], lw=1.4, label=v)
    cursor = axt.axvline(0, color="#ffffff", lw=0.8, alpha=0.6)
    axt.set_xlabel("step", color="#9aa"); axt.set_ylabel("population", color="#9aa")
    axt.tick_params(colors="#789"); [s.set_color("#334") for s in axt.spines.values()]
    axt.legend(facecolor=BG, edgecolor="#334", labelcolor="#cdd", fontsize=8, ncol=4, loc="upper right")
    stamp = fig.text(0.5, 0.985, "", color="#cdd", ha="center", va="top", family="monospace", fontsize=10)
    fig.text(0.5, 0.005, "colour = genome hash (fixed random projection of RNN weights) · trails = decaying feedback buffer · "
             "same seed, same physics, one switch per control", color="#678", ha="center", fontsize=7.5)
    w, h = fig.canvas.get_width_height()
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, "controls_side_by_side.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                           "-r", "30", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", mp4], stdin=subprocess.PIPE)
    for t in range(T):
        for v in VARIANTS:
            g, _ = gs[v]
            m = g["alive"][t]
            trails[v] *= 0.82
            trails[v] = splat(trails[v], g["pos"][t][m], g["colour"][t][m].astype(float) * 0.6, g["E"][t][m].astype(float), r=0)
            frame = np.maximum(field_rgb(g["field"][t]), trails[v])
            frame = splat(frame, g["pos"][t][m], g["colour"][t][m].astype(float), g["E"][t][m].astype(float), r=1)
            ims[v].set_data(np.clip(frame, 0, 1))
        cursor.set_xdata([t * 10, t * 10])
        stamp.set_text(f"step {t*10:5d}   " + "   ".join(f"{v}:{int(ms[v][0]['pop'][t])}" for v in VARIANTS))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        ff.stdin.write(buf.tobytes())
        if t in frames_png_at:
            fig.savefig(os.path.join(OUT, f"frame_{t*10:05d}.png"), facecolor=BG)
    ff.stdin.close(); ff.wait()
    plt.close(fig)
    return mp4

def knn_preservation(X, Y, k=10):
    """Mirror of mottled.projection.neighborhood_preservation (numpy, no sklearn)."""
    def knn(Z):
        d = ((Z[:, None] - Z[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
        return np.argsort(d, 1)[:, :k]
    a, b = knn(X), knn(Y)
    return np.array([len(set(a[i]) & set(b[i])) / k for i in range(len(X))])

def dashboard(gs, ms):
    g, prov = gs["full"]; T = g["alive"].shape[0]; t = T - 1
    steps = np.arange(T) * 10
    fig = plt.figure(figsize=(16, 10), dpi=110, facecolor=BG)
    def ax_(rect, title):
        ax = fig.add_axes(rect, facecolor=BG); ax.set_title(title, color="#cdd", fontsize=10, family="monospace", loc="left")
        ax.tick_params(colors="#789", labelsize=8); [s.set_color("#334") for s in ax.spines.values()]
        return ax
    # (a) controls time series
    for i, (key, lab) in enumerate([("pop", "population"), ("lineages", "distinct lineages (root ancestors)"),
                                    ("in_clusters", "fraction of agents in clusters ≥4")]):
        ax = ax_([0.05, 0.68 - i * 0.30, 0.27, 0.21], lab)
        for v in VARIANTS:
            ax.plot(steps[:len(ms[v][0][key])], ms[v][0][key], color=VCOL[v], lw=1.2, label=v)
        if i == 0:
            ax.legend(facecolor=BG, edgecolor="#334", labelcolor="#cdd", fontsize=7)
    ax_ = ax_
    # (b) latent view of hidden states, with Mottled-style fidelity
    m = g["alive"][t]; Hs = g["h"][t][m].astype(float); col = g["colour"][t][m].astype(float)
    Xc = Hs - Hs.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); Y = Xc @ Vt[:2].T
    ev = (S[:2] ** 2).sum() / (S ** 2).sum()
    pres = knn_preservation(Xc, Y, k=10)
    ax = ax_([0.37, 0.55, 0.28, 0.38], f"agent hidden states, step {t*10} (PCA)\n"
             f"var kept {ev:.0%} · kNN preserved {pres.mean():.2f} · low-fidelity {np.mean(pres < LOW_FIDELITY):.0%}")
    ok = pres >= LOW_FIDELITY
    ax.scatter(Y[ok, 0], Y[ok, 1], s=7, c=col[ok], linewidths=0)
    ax.scatter(Y[~ok, 0], Y[~ok, 1], s=10, c="#ffb000", marker="x", linewidths=0.6, label="✕ neighbourhood not preserved")
    ax.legend(facecolor=BG, edgecolor="#334", labelcolor="#cdd", fontsize=7, loc="lower right")
    # (c) cluster overlay
    ax = ax_([0.69, 0.55, 0.28, 0.38], "field + cluster overlay (≥4 agents, ring = cluster)")
    ax.imshow(field_rgb(g["field"][t]), extent=[0, 1, 1, 0])
    pos = g["pos"][t][m]; lab = clusters(pos, 1.9 * prov["config"]["r0"])
    ids, cnt = np.unique(lab, return_counts=True)
    ax.scatter(pos[:, 0], pos[:, 1], s=3, c=col, linewidths=0)
    for b in ids[cnt >= 4]:
        p = pos[lab == b]; c = p.mean(0); r = np.sqrt(((p - c) ** 2).sum(1)).max() + 0.01
        ax.add_patch(plt.Circle(c, r, fill=False, color="#ffffff", lw=0.6, alpha=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_xticks([]); ax.set_yticks([])
    # (d) lineage stack (Muller-ish)
    ax = ax_([0.37, 0.08, 0.28, 0.36], "top lineages over time (full) — share of population")
    roots = g["root"][:]; al = g["alive"][:]
    last = roots[t][al[t]]; top = [r for r, _ in sorted(zip(*np.unique(last, return_counts=True)), key=lambda x: -x[1])[:8]]
    stack = np.array([[(roots[s][al[s]] == r).mean() if al[s].any() else 0 for s in range(T)] for r in top])
    cols = []
    for r in top:
        idx = np.where((roots[t] == r) & al[t])[0][0]; cols.append(g["colour"][t][idx].astype(float))
    ax.stackplot(steps, stack, colors=cols, alpha=0.9)
    ax.set_xlabel("step", color="#9aa", fontsize=8)
    # (e) energy budget allocation
    ax = ax_([0.69, 0.08, 0.28, 0.36], "mean energy spend per agent per step, by action (full)")
    bud = ms["full"][1]; labs = ["basal", "move", "eat", "signal", "bond", "attack"]
    ax.stackplot(steps, bud.T, labels=labs, colors=["#445", "#7df9ff", "#a6ff7d", "#c39bff", "#ffd36f", "#ff5e5e"], alpha=0.9)
    ax.legend(facecolor=BG, edgecolor="#334", labelcolor="#cdd", fontsize=7, loc="upper left", ncol=3)
    ax.set_xlabel("step", color="#9aa", fontsize=8)
    fig.text(0.05, 0.965, "CA adaptive-agency MVP — analysis dashboard", color="#eef", fontsize=13, family="monospace")
    fig.text(0.05, 0.945, f"config {prov['config_sha256'][:12]} · code {prov['code_sha256'][:12]} · jax {prov['env']['jax']} · seed {prov['config']['seed']}",
             color="#678", fontsize=8, family="monospace")
    p = os.path.join(OUT, "agency_dashboard.png"); fig.savefig(p, facecolor=BG); plt.close(fig)
    return p, dict(var_kept=float(ev), knn_preserved=float(pres.mean()), low_fidelity=float(np.mean(pres < LOW_FIDELITY)))

if __name__ == "__main__":
    gs = {v: load(v) for v in VARIANTS}
    ms = {v: metrics(*gs[v]) for v in VARIANTS}
    os.makedirs(OUT, exist_ok=True)
    summ = {}
    for v in VARIANTS:
        mt = ms[v][0]
        summ[v] = {k: float(np.nanmean(mt[k][-50:])) for k in mt}
    dash, lat = dashboard(gs, ms)
    T = min(g["alive"].shape[0] for g, _ in gs.values())
    mp4 = video(gs, ms, frames_png_at={T // 4, T - 1})
    summ["latent_full_final"] = lat
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1)); print(dash, mp4)
