"""Run one sim10 config, compute per-window detector stats (docs/SIM10_DESIGN.md), save to OUT (outside the repo)."""
import sys, os, json, copy, time, resource
import numpy as np, pickle
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mvp.sim10 import V2, World, Cfg, GENES, gi, NS, NT, TAG, REC, aff, S as SIGCH, A as ACH, V as VCH

def cluster(gm, rng, cut=0.9):
    n = gm.shape[0]
    if n < 3: return np.zeros(n, int), gm[:1].copy()
    sub = rng.choice(n, min(n, 1000), replace=False)
    lab = fcluster(linkage(gm[sub], "average"), cut, "distance") - 1
    cents = np.stack([gm[sub][lab == k].mean(0) for k in range(lab.max() + 1)])
    d = ((gm[:, None, :] - cents[None]) ** 2).sum(-1)
    return d.argmin(1), cents

def corr_len(f):
    f = f - f.mean()
    if f.std() < 1e-15: return 0.0
    ac = np.fft.ifft2(np.abs(np.fft.fft2(f)) ** 2).real; ac /= ac[0, 0]
    G = f.shape[0]; yy, xx = np.mgrid[0:G, 0:G]; rr = np.sqrt(np.minimum(xx, G - xx) ** 2 + np.minimum(yy, G - yy) ** 2)
    for R in np.arange(1, G // 2):
        if ac[(rr >= R - 0.5) & (rr < R + 0.5)].mean() < 1 / np.e: return float(R)
    return float(G // 2)

def window_stats(w, lab_all, cents, rng):
    L = w.log; a = np.flatnonzero(w.alive); K = cents.shape[0]
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    bf = np.bincount(cat(L["birth_free"]).astype(int).clip(0), minlength=K)[:K]
    bi = np.bincount(cat(L["birth_in"]).astype(int).clip(0), minlength=K)[:K]
    labw = w.lab.copy(); valid = labw >= 0
    fs = np.bincount(labw[valid], L["free_steps"][valid], K)[:K]; ins = np.bincount(labw[valid], L["in_steps"][valid], K)[:K]
    fin = np.bincount(labw[valid], L["field_in"][valid], K)[:K]
    flow = np.zeros((K, K))   # flow[src, dst]
    for key in ("eng", "drain"):
        if L[key]:
            e = np.concatenate(L[key]); e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)]
            np.add.at(flow, (e[:, 1].astype(int), e[:, 0].astype(int)), e[:, 2])
    flow_nd = flow.copy(); np.fill_diagonal(flow_nd, 0)   # post-hoc variant: exclude within-cluster feeding
    TLnd = np.ones(K)
    for _ in range(50):
        tot = fin + flow_nd.sum(0)
        TLnd = 1 + np.where(tot > 0, (flow_nd * TLnd[:, None]).sum(0) / np.maximum(tot, 1e-300), 0)
    TL = np.ones(K)
    for _ in range(50):
        tot = fin + flow.sum(0)
        TL = 1 + np.where(tot > 0, (flow * TL[:, None]).sum(0) / np.maximum(tot, 1e-300), 0)
    inf_t = np.concatenate(L["inf_tags"]) if L["inf_tags"] else np.zeros((0, 2 * NT))
    if len(inf_t) > 200: inf_t = inf_t[rng.choice(len(inf_t), 200, replace=False)]
    # aggregates (D3) at window end, free agents
    G = w.c.G; fr = a[w.host[a] < 0]; cell = w.x[fr] * G + w.y[fr]; ad = w.g[fr, gi["adh"]] > 0.5
    nadh = np.bincount(cell[ad], minlength=G * G); aggc = np.flatnonzero(nadh >= 4)
    inagg = ad & np.isin(cell, aggc)
    sds = [w.p[fr[inagg & (cell == cc)]].std() for cc in aggc]
    bp = np.concatenate(L["birth_p"]) if L["birth_p"] else np.zeros((0, 3))
    bpa = bp[np.isin(bp[:, 0].astype(int), aggc) & (bp[:, 2] > 0.5)]
    pin = w.p[fr[inagg]]
    pc = np.bincount(lab_all, minlength=K) / max(len(lab_all), 1)
    H = -(pc[pc > 0] * np.log(pc[pc > 0])).sum()
    return dict(t=w.t, N=int(a.size), n_inside=int((w.host[a] >= 0).sum()), K=int(K), expH=float(np.exp(H)),
                abund=pc.tolist(), bf=bf.tolist(), bi=bi.tolist(), fs=fs.tolist(), ins=ins.tolist(),
                TL=TL.tolist(), TLnd=TLnd.tolist(), self_flow=float(np.trace(flow)), fin=fin.tolist(), flow_tot=float(flow.sum()), n_inf=int(sum(len(x) for x in L["inf"])),
                n_eng=int(sum(len(x) for x in L["eng"])),
                agg_frac=float(inagg.sum() / max(a.size, 1)), agg_sd=float(np.mean(sds)) if sds else float("nan"),
                agg_birth_lowp=float((bpa[:, 1] < 0.5).mean()) if len(bpa) else float("nan"),
                agg_highp_frac=float((pin >= 0.5).mean()) if pin.size else float("nan"), n_agg_births=int(len(bpa)),
                mem=float(w.g[a, gi["mem"]].mean()), gene_mean=w.g[a, :NS].mean(0).tolist(),
                sig_rel=L["sig_rel"], sig_rt=L["sig_rt"], sig_dec=L["sig_dec"],
                sig_mean=float(w.F[SIGCH].mean()), V_mean=float(w.F[VCH].mean()), A_mean=float(w.F[ACH].mean()),
                sig_corr=corr_len(w.F[SIGCH]), sig_net=L["sig_net"], sig_work=L["sig_work"],
                sig_halflife=sig_hl(w, L), V_var=float(w.F[VCH].var()), V_spots=int(ndimage.label(w.F[VCH] > 0.1)[1]),
                gene_sd=w.g[a, :NS].std(0).tolist() if a.size else [], sigma_win=float(np.sum([r_[3] for r_ in w.resid[-w.c.W:]])),
                resid_win=float(np.max([max(abs(r_[1]), abs(r_[2])) for r_ in w.resid[-w.c.W:]]))), cents, inf_t

def sig_hl(w, L):
    stock = w.F[SIGCH].sum()
    if stock <= 0: return float("nan")
    rate = (L["sig_dec"] + L["sig_rt"]) / w.c.W / stock + w.c.f * w.c.inflow
    return float(np.log(2) / rate) if rate > 0 else float("inf")

def ko_test(w, T=200):
    out = {}
    for ko in (0, 1):
        w2 = copy.deepcopy(w); w2.c = type(w.c)(**{**w.c.__dict__, "rt_sig_ko": ko}); w2.rng = np.random.default_rng(777)
        sm, cl = [], []; w2.new_window(); st = []
        for t in range(T):
            w2.step(); st.append(w2.F[SIGCH].sum())
            if t >= T - 50: sm.append(w2.F[SIGCH].mean()); cl.append(corr_len(w2.F[SIGCH]))
        L = w2.log; rate = (L["sig_dec"] + L["sig_rt"]) / T / max(np.mean(st), 1e-300) + w2.c.f * w2.c.inflow
        out[ko] = (float(np.mean(sm)), float(np.mean(cl)), float(np.log(2) / rate), L["sig_net"], L["sig_work"], L["sig_rt"])
    return dict(sig_intact=out[0][0], sig_ko=out[1][0], corr_intact=out[0][1], corr_ko=out[1][1],
                hl_intact=out[0][2], hl_ko=out[1][2], rt_net_energy_intact=out[0][3], rt_work_intact=out[0][4], rt_flux_intact=out[0][5])

def main(name, outdir, **kw):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    if kw.pop("preset", 0) == 2: kw = {**V2, **kw}
    cfg = Cfg(**kw); ck = f"{outdir}/{name}.ckpt"
    if os.path.exists(ck):
        w, rng, cents, wins, centl, inft, snaps = pickle.load(open(ck, "rb")); print(name, "resumed at", w.t, flush=True)
    else:
        w = World(cfg); rng = np.random.default_rng(cfg.seed + 99)
        a = np.flatnonzero(w.alive); lab, cents = cluster(w.g[a], rng); w.lab[:] = -1; w.lab[a] = lab
        wins, centl, inft, snaps = [], [], [], {}
    while w.t < cfg.steps:
        w.step()
        if w.t % 5000 == 0 and w.t < cfg.steps:
            pickle.dump((w, rng, cents, wins, centl, inft, snaps), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
        if w.t % cfg.W == 0:
            a = np.flatnonzero(w.alive)
            lab_new, cents_new = cluster(w.g[a], rng) if a.size else (np.zeros(0, int), np.zeros((1, w.g.shape[1])))
            st, _, it = window_stats(w, lab_new, cents, rng)   # events were labelled with previous centroids
            wins.append(st); centl.append(cents); inft.append(it)
            w.lab[:] = -1; w.lab[a] = lab_new; cents = cents_new; w.new_window()
            if w.t in (cfg.steps // 2, cfg.steps):
                s = w.snapshot(); s["F"] = w.F.copy(); s["lab"] = lab_new; snaps[w.t] = s
    ko = ko_test(w) if w.alive.any() else None
    R = np.array(w.resid)
    res = dict(name=name, cfg=cfg.__dict__, windows=wins, ko=ko, runtime_s=time.time() - t0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               thermo=dict(max_abs_rM=float(np.abs(R[:, 1]).max()), max_abs_rE=float(np.abs(R[:, 2]).max()),
                           min_sigma_step=float(R[:, 3].min()), min_process_diss=float(R[:, 4].min()),
                           total_sigma=float(R[:, 3].sum()), total_heat=float(R[:, 5].sum()), steps=int(len(R))))
    json.dump(res, open(f"{outdir}/{name}.json", "w"))
    np.savez_compressed(f"{outdir}/{name}.npz", cents=np.array(centl, dtype=object), inf=np.array(inft, dtype=object),
                        resid=R, snaps=np.array([snaps], dtype=object))
    if os.path.exists(ck): os.remove(ck)
    print(name, "done", round(time.time() - t0, 1), "s", res["thermo"], flush=True)

if __name__ == "__main__":
    name, outdir = sys.argv[1], sys.argv[2]
    kw = {k: (float(v) if "." in v else int(v)) for k, v in (a.split("=") for a in sys.argv[3:])}
    main(name, outdir, **kw)
