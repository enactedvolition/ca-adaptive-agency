"""Per-run metrics for sim2 (host-side numpy). Every estimator here is a simplified,
labelled proxy; see ARCHITECTURE.md §11 for definitions, citations and confounds."""
from __future__ import annotations
import numpy as np

LAM_BINS = np.linspace(-1.5, 1.0, 51)
NEAR, ORD, CHA = 0.05, -0.1, 0.1          # thresholds from Evelyn's NDE near-criticality page (portfolio/index.html)

def lam_stats(lam):
    lam = lam[np.isfinite(lam)]
    if len(lam) < 5:
        return dict(lam_med=np.nan, lam_absmed=np.nan, f_near=np.nan, f_ord=np.nan, f_cha=np.nan, n_lam=len(lam))
    return dict(lam_med=float(np.median(lam)), lam_absmed=float(np.median(np.abs(lam))),
                f_near=float(np.mean(np.abs(lam) <= NEAR)), f_ord=float(np.mean(lam < ORD)),
                f_cha=float(np.mean(lam > CHA)), n_lam=int(len(lam)))

def _pca(X, k):
    X = X - X.mean(0); sd = X.std(0); X = X[:, sd > 1e-8] / sd[sd > 1e-8]
    if X.shape[1] == 0:
        return np.zeros((len(X), 1))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:min(k, X.shape[1])].T

def _logdet(Z, ridge=1e-3):
    if Z.shape[1] == 0:
        return 0.0
    C = np.cov(Z, rowvar=False).reshape(Z.shape[1], Z.shape[1])
    C = C + ridge * np.eye(len(C)) * max(np.trace(C) / len(C), 1e-12)
    return np.linalg.slogdet(C)[1]

def gcmi(X, Y, Z=None):
    """Gaussian (conditional) mutual information in nats."""
    if Z is None or Z.shape[1] == 0:
        return 0.5 * (_logdet(X) + _logdet(Y) - _logdet(np.hstack([X, Y])))
    return 0.5 * (_logdet(np.hstack([X, Z])) + _logdet(np.hstack([Y, Z])) - _logdet(Z) - _logdet(np.hstack([X, Y, Z])))

def bc_gcmi(X, Y, Z=None, rng=None, n_perm=5):
    """Bias-corrected: raw minus mean over permutations of Y's rows (breaks the X-Y pairing only)."""
    rng = rng or np.random.default_rng(0)
    raw = gcmi(X, Y, Z)
    null = np.mean([gcmi(X, Y[rng.permutation(len(Y))], Z) for _ in range(n_perm)]) if Z is None else \
        np.mean([gcmi(X, *(lambda p: (Y[p], Z[p]))(rng.permutation(len(Y)))) for _ in range(n_perm)])
    return float(raw - null)

def pair_snapshots(snaps):
    """Match agents across consecutive dense snapshots by uid -> list of (a, b) index arrays."""
    pairs = []
    for s0, s1 in zip(snaps[:-1], snaps[1:]):
        common, i0, i1 = np.intersect1d(s0["uid"], s1["uid"], return_indices=True)
        pairs.append((s0, s1, i0, i1))
    return pairs

def info_metrics(snaps, ext, k=4, min_n=60):
    """Simplified Krakauer et al. 2020 individuality on agent hidden states, plus a sensorimotor
    (empowerment-proxy) information, pooled over consecutive snapshot pairs (dt = snapshot stride).
      S = PCA_k(h), E = PCA_k(external senses), A_act = PCA_k(actions)
      A* = I(S'; S)   A = I(S'; S | E)   nC = I(S'; E | S)   emp = I(A_act; X' | X)
    Gaussian estimators, permutation bias-corrected. Agent-level, NOT cluster-level."""
    H0, H1, X0, X1, A0 = [], [], [], [], []
    for s0, s1, i0, i1 in pair_snapshots(snaps):
        H0.append(s0["h"][i0]); H1.append(s1["h"][i1]); X0.append(s0["x"][i0][:, ext]); X1.append(s1["x"][i1][:, ext]); A0.append(s0["act"][i0])
    if not H0 or sum(len(a) for a in H0) < min_n:
        return dict(A_star=np.nan, A_col=np.nan, nC=np.nan, emp=np.nan, n_pairs=0)
    H0, H1, X0, X1, A0 = [np.concatenate(v).astype(np.float64) for v in (H0, H1, X0, X1, A0)]
    allH = _pca(np.vstack([H0, H1]), k); S, S1 = allH[:len(H0)], allH[len(H0):]
    allX = _pca(np.vstack([X0, X1]), k); E, E1 = allX[:len(X0)], allX[len(X0):]
    Aa = _pca(A0, k)
    rng = np.random.default_rng(1)
    return dict(A_star=bc_gcmi(S1, S, None, rng), A_col=bc_gcmi(S1, S, E, rng), nC=bc_gcmi(S1, E, S, rng),
                emp=bc_gcmi(E1, Aa, E, rng), n_pairs=int(len(H0)))

def heritability(snap, min_age=100):
    """Parent-offspring regression of reproductive rate: corr(parent's rate at the child's birth, child's rate)."""
    age = snap["age"]; ok = (age >= min_age) & np.isfinite(snap["prate"])
    if ok.sum() < 15:
        return np.nan, int(ok.sum())
    a, b = snap["prate"][ok], snap["n_off"][ok] / age[ok]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())

def simpson(root):
    if len(root) == 0:
        return np.nan
    _, c = np.unique(root, return_counts=True); p = c / c.sum()
    return float((p**2).sum())

# ---------------- dark-room classification
def f_var(amp, W, theta, k_theta, k_wall):
    """Analytic stationary variance of the OU fluctuation field at wall density W."""
    th = theta + k_theta * W / (1.0 + W)
    return (amp * np.exp(-k_wall * W)) ** 2 * 2 * theta / (2 * th - th**2)

REGIMES = ["explore", "settled-noisy", "dark: natural", "dark: other-built", "dark: self-built"]

def regimes(snap, cfg, env_var, Amap, v_term):
    """Per-agent behavioural regime.
      resident = residence >= 100 steps and speed < 0.2 * terminal speed
      quiet    = analytic local fluctuation variance < 0.5 * map-mean undamped variance
      dark: natural    resident & quiet even with walls removed (passive dark-rooming)
      dark: self-built resident & quiet, but NOT quiet if this agent's own recent deposits are removed (counterfactual)
      dark: other-built resident & quiet only because of walls others left (kin or not)
    Returns int codes (-1 = undefined when env_var == 0)."""
    n = len(snap["uid"])
    resident = (snap["res"] >= 100) & (snap["spd"] < 0.2 * v_term)
    code = np.where(resident, 1, 0)
    if env_var <= 0:
        return np.where(resident, -1, 0)
    amp = env_var * Amap[snap["cx"], snap["cy"]]
    gref = np.mean(f_var(env_var * Amap, 0.0, cfg.theta, cfg.k_theta, cfg.k_wall))
    W = snap["W_here"] * snap["build_eff"]; own = snap["own"] * snap["build_eff"]
    v_now = f_var(amp, W, cfg.theta, cfg.k_theta, cfg.k_wall)
    v_nat = f_var(amp, 0.0, cfg.theta, cfg.k_theta, cfg.k_wall)
    v_noself = f_var(amp, np.maximum(W - own, 0), cfg.theta, cfg.k_theta, cfg.k_wall)
    quiet = v_now < 0.5 * gref
    code = np.where(resident & ~quiet, 1, code)
    code = np.where(resident & quiet & (v_nat < 0.5 * gref), 2, code)
    code = np.where(resident & quiet & (v_nat >= 0.5 * gref) & (v_noself >= 0.5 * gref), 4, code)
    code = np.where(resident & quiet & (v_nat >= 0.5 * gref) & (v_noself < 0.5 * gref), 3, code)
    return code
