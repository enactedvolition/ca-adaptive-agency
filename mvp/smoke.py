"""Smoke test: tiny sim4 world on CPU (<1 min), checks finiteness, writes one frame PNG.

    python -m mvp.smoke [--steps 100] [--out out_smoke] [--demo]

--demo runs a bigger/longer world and also writes an animated GIF (used by run.sh / run.bat).
"""
from __future__ import annotations
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", ""))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def frame(ax, st, cfg, colour, t):
    ax.clear()
    ax.imshow(np.asarray(st["Rf"]).T, origin="lower", extent=[0, cfg.L, 0, cfg.L], cmap="Greens", vmin=0, vmax=1.2)
    W = np.asarray(st["Wl"]).T
    if W.max() > 0: ax.imshow(W, origin="lower", extent=[0, cfg.L, 0, cfg.L], cmap="Greys", alpha=0.35)
    al = np.asarray(st["alive"]); p = np.asarray(st["pos"])[al]
    ax.scatter(p[:, 0], p[:, 1], s=6, c=np.asarray(colour)[al], edgecolors="none")
    ax.set_xlim(0, cfg.L); ax.set_ylim(0, cfg.L); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {t}  agents {int(al.sum())}", fontsize=9)

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import jax, sim4
    if a.demo:
        cfg = sim4.Cfg(N=1024, L=0.7, G=48); steps = a.steps or 1500; every = 25; out = a.out or "out_demo"
    else:
        cfg = sim4.Cfg(N=128, L=0.25, G=16); steps = a.steps or 100; every = steps; out = a.out or "out_smoke"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    init, chunk_one, colour_fn, consts = sim4.make(cfg)
    prm = sim4.params(env_var=0.35)
    st = jax.jit(init)(jax.random.PRNGKey(a.seed), prm)
    step = jax.jit(lambda s, p: chunk_one(s, p, every)); col = jax.jit(colour_fn)
    fig, ax = plt.subplots(figsize=(5, 5)); frames = []
    for t in range(every, steps + 1, every):
        st = step(st, prm)
        if a.demo:
            frame(ax, st, cfg, col(st["W"]), t); fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    E = np.asarray(st["E"]); al = np.asarray(st["alive"])
    ok = bool(np.isfinite(E).all() and np.isfinite(np.asarray(st["h"])).all() and al.sum() > 0)
    frame(ax, st, cfg, col(st["W"]), steps); png = os.path.join(out, "frame.png"); fig.savefig(png, dpi=110, bbox_inches="tight")
    msg = f"{'SMOKE OK' if ok else 'SMOKE FAIL'}: {steps} steps, {int(al.sum())} agents alive, backend={jax.default_backend()}, {time.time()-t0:.1f}s -> {png}"
    if a.demo and frames:
        from PIL import Image
        gif = os.path.join(out, "demo.gif")
        Image.fromarray(frames[0]).save(gif, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]], duration=80, loop=0)
        msg += f", {gif}"
    print(msg)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
