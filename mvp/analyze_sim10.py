"""Apply the pre-specified sim10 detectors (docs/SIM10_DESIGN.md) to runs in RUNS; write tables + figures."""
import sys, os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mvp.sim10 import aff, NT, GENES, NS, SPN
RUNS = sys.argv[1]; FIG = sys.argv[2]; os.makedirs(FIG, exist_ok=True)
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{RUNS}/*.json"))}

def d1(ws):
    hits = 0; ex = []
    half = len(ws) // 2
    for wi, w in enumerate(ws):
        for k in range(len(w["bf"])):
            if wi < half or w["abund"] and k < len(w["abund"]) and False: pass
            fs, ins, bf, bi = w["fs"][k], w["ins"][k], w["bf"][k], w["bi"][k]
            if ins >= 200 and fs >= 200 and bi > 0:
                rf, ri = bf / fs, bi / ins
                if rf <= 0.1 * ri and wi >= half: hits += 1; ex.append((w["t"], k, rf, ri))
    return hits >= 3, hits, ex[:3]

def d2(name):
    z = np.load(f"{RUNS}/{name}.npz", allow_pickle=True); inf = list(z["inf"])
    wins = [i for i, x in enumerate(inf) if len(x) >= 5]
    if len(wins) < 12: return False, dict(n_inf_windows=len(wins))
    cent = {i: inf[i][:, NT:].mean(0) for i in wins}
    disp = float(np.linalg.norm(cent[wins[-1]] - cent[wins[0]]))
    def Aab(i, j):
        P, H = inf[i][:, :NT], inf[j][:, NT:]
        return float(aff(P[:, None, :], H[None, :, :], 0.5).mean())
    A0 = np.mean([Aab(i, i) for i in wins]); fut = [Aab(i, i + 8) for i in wins if (i + 8) in cent]
    past = [Aab(i, i - 8) for i in wins if (i - 8) in cent]
    A8 = np.mean(fut) if fut else np.nan
    return bool(disp >= 0.3 and (A0 - A8) >= 0.05), dict(n_inf_windows=len(wins), host_tag_disp=disp, A0=A0, A_fut8=A8,
                                                        A_past8=np.mean(past) if past else np.nan)

def d3(ws):
    h = ws[len(ws) // 2:]
    fa = np.nanmean([w["agg_frac"] for w in h]); sd = np.nanmean([w["agg_sd"] for w in h])
    bl = np.nanmean([w["agg_birth_lowp"] for w in h]); hp = np.nanmean([w["agg_highp_frac"] for w in h])
    return bool(fa >= 0.1 and sd >= 0.2 and bl >= 0.7 and hp >= 0.25), dict(agg_frac=fa, within_sd_p=sd, births_lowp=bl, highp_frac=hp)

def d4(ws):
    h = ws[len(ws) // 2:]; mx = []
    for w in h:
        ab = np.array(w["abund"]); TL = np.array(w["TL"])[: len(ab)]
        mx.append(TL[ab[: len(TL)] >= 0.02].max() if (ab[: len(TL)] >= 0.02).any() else 1)
    return bool(sum(m >= 2.5 for m in mx) >= 3), dict(maxTL_median=float(np.median(mx)), maxTL_max=float(np.max(mx)))

def d5(name, ws):
    z = np.load(f"{RUNS}/{name}.npz", allow_pickle=True); C = list(z["cents"])
    h = ws[len(ws) // 2:]; eh = float(np.median([w["expH"] for w in h]))
    # tracks: chain nearest centroids d<0.9 across consecutive windows
    W = ws[0]["t"]; tracks = {k: 1 for k in range(len(C[0]))}; long = 0; cur = {k: k for k in range(len(C[0]))}
    life = {k: 1 for k in range(len(C[0]))}
    for i in range(1, len(C)):
        nl = {}
        for j in range(len(C[i])):
            d = np.linalg.norm(C[i - 1] - C[i][j], axis=1); k = d.argmin()
            nl[j] = life.get(k, 0) + 1 if d[k] < 0.9 else 1
        life = nl
        long = max(long, sum(v * W >= 1000 for v in life.values()))
    return bool(eh >= 3 and long >= 3), dict(expH_med=eh, n_tracks_ge1000=int(long))

rows = []
for n, r in R.items():
    ws = r["windows"]
    if not ws: continue
    row = dict(run=n, N_end=ws[-1]["N"], inside_end=ws[-1]["n_inside"], **{f"th_{k}": v for k, v in r["thermo"].items()},
               runtime=r["runtime_s"], rss=r["maxrss_mb"])
    if n.startswith(("closed", "agentfree")):
        row.update(A_end=ws[-1]["A_mean"], V_end=ws[-1]["V_mean"]); rows.append(row); continue
    f1, h1, ex = d1(ws); f2, i2 = d2(n); f3, i3 = d3(ws); f4, i4 = d4(ws); f5, i5 = d5(n, ws)
    row.update(D1=f1, D1_hits=h1, D2=f2, **{f"D2_{k}": v for k, v in i2.items()}, D3=f3, **{f"D3_{k}": v for k, v in i3.items()},
               D4=f4, **{f"D4_{k}": v for k, v in i4.items()}, D5=f5, **{f"D5_{k}": v for k, v in i5.items()},
               mem2=float(np.mean([w["mem"] for w in ws[len(ws) // 2:]])))
    ko = r["ko"]
    if ko:
        row.update(D7_sig_ratio=ko["sig_ko"] / max(ko["sig_intact"], 1e-300), D7_corr_ratio=ko["corr_ko"] / max(ko["corr_intact"], 1e-9))
    rows.append(row)
json.dump(rows, open(f"{FIG}/detectors.json", "w"), indent=1, default=float)
# D6 paired: full_in1.0 vs meminert by seed
d6 = [(next(x["mem2"] for x in rows if x["run"] == f"full_in1.0_s{s}"), next(x["mem2"] for x in rows if x["run"] == f"meminert_s{s}"))
      for s in range(3) if any(x["run"] == f"full_in1.0_s{s}" for x in rows) and any(x["run"] == f"meminert_s{s}" for x in rows)]
if d6:
    dd = np.array([a - b for a, b in d6]); m = dd.mean(); se = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
    print(f"D6 mem full-inert per seed {dd.round(3).tolist()} mean {m:+.3f} CI [{m-4.303*se:+.3f},{m+4.303*se:+.3f}]")
for x in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})

# ---- figures ----
cols = plt.cm.tab10.colors
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for n, r in R.items():
    if n.startswith("agentfree"): continue
    ws = r["windows"]; t = [w["t"] for w in ws]
    c = cols[hash(n.split("_s")[0]) % 10]
    ax[0, 0].plot(t, [w["expH"] for w in ws], color=c, alpha=.7, label=n.split("_s")[0] if n.endswith("s0") else None)
    ax[0, 1].plot(t, [w["N"] for w in ws], color=c, alpha=.7)
    ax[1, 0].plot(t, [w["n_inside"] for w in ws], color=c, alpha=.7)
    ax[1, 1].plot(t, [w["agg_frac"] for w in ws], color=c, alpha=.7)
for a_, tt in zip(ax.ravel(), ["genotype diversity exp(H)", "agents alive", "agents inside a host", "fraction in adhesive aggregates"]): a_.set_title(tt)
ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/diversity_time.png", dpi=110)
# field snapshot for full_in1.0_s0
for n in ("full_in1.0_s0", "closed_s0", "agentfree_s0"):
    if not os.path.exists(f"{RUNS}/{n}.npz"): continue
    sn = np.load(f"{RUNS}/{n}.npz", allow_pickle=True)["snaps"][0]
    if not sn: continue
    s = sn[max(sn)]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    for a_, ch in zip(ax[:4], [0, 1, 3, 4]):
        im = a_.imshow(s["F"][ch], cmap="magma"); a_.set_title(f"{SPN[ch]} t={s['t']}"); plt.colorbar(im, ax=a_, fraction=.046)
    if len(s["x"]):
        G = s["F"].shape[1]; jit = np.random.default_rng(0).random((2, len(s["x"]))) * .8 - .4
        ax[4].imshow(s["F"][0], cmap="gray", alpha=.4)
        ax[4].scatter(s["y"] + jit[1], s["x"] + jit[0], s=4, c=[cols[l % 10] for l in s["lab"]], marker="o")
        ax[4].scatter((s["y"] + jit[1])[s["host"]], (s["x"] + jit[0])[s["host"]], s=10, facecolors="none", edgecolors="w")
    ax[4].set_title("agents by genotype cluster (white ring = inside host)")
    fig.tight_layout(); fig.savefig(f"{FIG}/field_{n}.png", dpi=100); plt.close("all")
# thermo residual + closed-system relaxation
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for n in R:
    z = np.load(f"{RUNS}/{n}.npz", allow_pickle=True)["resid"]
    ax[0].semilogy(z[:, 0], np.abs(z[:, 2]) + 1e-18, lw=.5, alpha=.5)
    if n.startswith("closed"):
        ws = R[n]["windows"]; ax[1].plot([w["t"] for w in ws], [w["N"] for w in ws], label=n)
        ax[2].plot([w["t"] for w in ws], [w["A_mean"] for w in ws], label=n + " A"); ax[2].plot([w["t"] for w in ws], [w["V_mean"] for w in ws], "--")
ax[0].set_title("|energy ledger residual| (relative), all runs"); ax[1].set_title("closed system: agents alive"); ax[2].set_title("closed system: mean A (solid), V (dashed)")
ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/thermo_checks.png", dpi=110)
