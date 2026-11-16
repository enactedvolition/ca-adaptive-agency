"""Exp 15 pre-check (observer only, no new sim): is the energy sense already used in the evolved Exp 13 sim9 populations?
For every live agent on the t=10000 sw_full forks: es = d eat / d sensed energy (the sim11 formula, from the fork's own h, act, W_ih row 8, W_ho col 2),
vs the same quantity with the W_ih row-8 vector shuffled across agents (breaks its link to the agent's own readout), and the cross-agent
correlation of the eat action with true energy. Also the row-8 weight norm relative to the other input rows."""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim9
FORKS = os.environ.get("EXP15_PRIOR_FORKS", "/workspace/ca-agency-viz/mvp/runs13")
cfg = sim9.Cfg(N=2048, L=1.0, G=64, claim_mode="mass"); lay, P, _ = sim9.layout(cfg.H); H = cfg.H
off = {n: (o, sh) for n, sh, o, z in lay}
out = {}
for arm in ("sw_full", "sw_mg2"):
    rows = []
    for s in range(9):
        z = np.load(os.path.join(FORKS, f"fork13_{arm}_s{s}_t10000.npz")); al = z["alive"]; W = z["W"][al].astype(np.float64)
        Wih = W[:, off["W_ih"][0]:off["W_ih"][0] + 16 * H].reshape(-1, 16, H); Who = W[:, off["W_ho"][0]:off["W_ho"][0] + H * 8].reshape(-1, H, 8)
        h = z["h"][al].astype(np.float64); a = z["act"][al].astype(np.float64); E = z["E"][al].astype(np.float64)
        f = lambda w8: 0.5 * (1 - a[:, 2] ** 2) * (((1 - h ** 2) * w8) * Who[:, :, 2]).sum(1)
        es = f(Wih[:, 8, :]); rng = np.random.default_rng(s); es_sh = f(Wih[rng.permutation(len(es)), 8, :])
        norms = np.linalg.norm(Wih, axis=2).mean(0)
        eat = (a[:, 2] + 1) / 2
        rows.append(dict(seed=s, n=int(al.sum()), es_mean=es.mean(), es_abs=np.abs(es).mean(), es_sh_mean=es_sh.mean(), es_sh_abs=np.abs(es_sh).mean(),
                         row8_norm=norms[8], other_rows_norm=float(np.delete(norms, [8, 14]).mean()), corr_eat_E=float(np.corrcoef(eat, E)[0, 1]),
                         frac_neg=float((es < 0).mean()), frac_neg_sh=float((es_sh < 0).mean())))
    out[arm] = rows
    keys = [k for k in rows[0] if k not in ("seed", "n")]
    print(arm, {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}, flush=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs15"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs15", "prior_check.json"), "w"), indent=1)
