"""sim v3 = sim v2 + replay logging (identical dynamics; extra observer fields only).

Added state: gpar (genetic parent uid = genome donor), anc (ancestor in the population at the last
logging boundary; reset by the caller), l0s/l0n (lambda accumulator from birth, no burn-in), lstep
(this step's log tangent stretch), cstep (this step's energy spend), c_int/c_cost/c_act/c_eat/c_build
(lifetime accumulators). Nothing added feeds back into the dynamics.

sim v2: stateful, agent-coupled environment + per-brain criticality + dark-room observables.

Off-grid agents (torus of side L), each a random Elman RNN (15 senses -> 16 hidden -> 8 actions).
Environment (G x G grids, all coupled to agents):
  R   resource: logistic regrowth  dR = r R (1 - R/K(x)) (1 + fert P) + seed (K - R)  - harvest
  F   fluctuation ("weather"): OU noise, amplitude env_var * A(x); richer patches are noisier.
      F corrupts the sensed resource, buffets motion, and makes harvest variable.
  Wl  walls/structure: deposited by the BUILD action; damp F locally (amplitude * exp(-kW Wl),
      faster OU decay) and add drag. This is the niche-construction channel for dark rooms.
  P   pheromone/waste: deposited by metabolism (waste) and |signal|; diffuses, decays;
      fertilises regrowth, toxic at high levels (cost c_tox P^2). Agents sense P and grad P.
Observer-side instruments (do not drive behaviour):
  - Lyapunov: tangent vector pushed through the recurrent step with jax.jvp under the agent's
    actual inputs, renormalised each step; lambda_max = mean log growth (after 10-step burn-in).
  - mean-field gain  sigma^2 = (1/H) ||J||_F^2 (perturbation-spread / branching proxy; ~1 at the edge).
  - surprisal: an NLMS linear readout from the agent's own hidden state predicting its next
    external input; Gaussian surprisal of its prediction error (running variance), EMA.
  - input variance / Gaussian sensory entropy (EMA), speed, residence time, own-wall deposit.
Controls are traced params (one compile): cost_mul (no_cost=0), mut_sigma (no_mut=0),
shuffle (heredity broken), feedback (0 = static env: no depletion, no P/W effects), build_eff (0 = walls inert).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import jax, jax.numpy as jnp

@dataclass(frozen=True)
class Cfg:
    N: int = 512
    H: int = 16
    L: float = 0.5
    G: int = 32
    R: float = 0.03
    M: int = 12                 # max agents per neighbour cell (overflow counted)
    n_init_frac: float = 0.25
    r0: float = 0.008
    a_max: float = 0.002
    damp: float = 0.8
    k_rep: float = 0.3
    k_adh: float = 0.15
    E0: float = 1.0
    E_rep: float = 1.6
    E_cap: float = 4.0
    c_move: float = 0.01
    c_eat: float = 0.002
    c_sig: float = 0.001
    c_bond: float = 0.001
    c_att: float = 0.004
    c_build: float = 0.003
    c_tox: float = 0.02
    harvest: float = 0.12
    share_rate: float = 0.05
    attack_rate: float = 0.08
    attack_eff: float = 0.5
    regrow: float = 0.03
    seed_regrow: float = 0.0005
    fert: float = 0.5
    n_patches: int = 4
    patch_sigma: float = 0.08
    theta: float = 0.05          # OU relaxation
    k_theta: float = 0.2         # max extra relaxation from walls (saturating in W)
    k_wall: float = 2.0          # amplitude damping per unit wall
    wall_decay: float = 0.002
    wall_dep: float = 0.02
    wall_drag: float = 1.0
    p_decay: float = 0.02
    p_diff: float = 0.1
    p_dep_sig: float = 0.01
    waste_frac: float = 0.5
    k_push: float = 0.02
    noise_sense: float = 1.0
    spawn_prob: float = 0.1
    min_pop_frac: float = 0.04
    rescue_per_step: int = 4
    g_lo: float = 0.3            # prior over recurrent gain g (W_hh ~ g/sqrt(H) N(0,1))
    g_hi: float = 6.0
    lyap_burn: int = 10
    pred_lr: float = 0.05

I_DIM, A_DIM = 15, 8
EXT = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])   # external sensory channels
C_EXT = len(EXT)
PARAM_KEYS = ("c_basal", "mut_sigma", "env_var", "cost_mul", "shuffle", "feedback", "build_eff")

def params(c_basal=0.004, mut_sigma=0.05, env_var=0.25, cost_mul=1.0, shuffle=0.0, feedback=1.0, build_eff=1.0):
    return {k: jnp.float32(v) for k, v in dict(c_basal=c_basal, mut_sigma=mut_sigma, env_var=env_var, cost_mul=cost_mul,
                                                shuffle=shuffle, feedback=feedback, build_eff=build_eff).items()}

def layout(H):
    shapes = [("W_ih", (I_DIM, H)), ("W_hh", (H, H)), ("b_h", (H,)), ("W_ho", (H, A_DIM)), ("b_o", (A_DIM,))]
    sc = {"W_ih": 1.5 / np.sqrt(I_DIM), "W_hh": 1.0 / np.sqrt(H), "b_h": 0.1, "W_ho": 1.5 / np.sqrt(H), "b_o": 0.3}
    off, lay, s = 0, [], []
    for n, sh in shapes:
        z = int(np.prod(sh)); lay.append((n, sh, off, z)); s.append(np.full(z, sc[n])); off += z
    return lay, off, jnp.asarray(np.concatenate(s), jnp.float32)

def unpack(W, lay):
    return {n: W[:, o:o + z].reshape((W.shape[0],) + sh) for n, sh, o, z in lay}

def make(cfg: Cfg):
    lay, P, scale = layout(cfg.H)
    N, G, L, H, M = cfg.N, cfg.G, cfg.L, cfg.H, cfg.M
    hh_off = [o for n, sh, o, z in lay if n == "W_hh"][0]
    cs = L / G
    C = max(3, int(np.floor(L / cfg.R)))
    rng = np.random.default_rng(1234)
    colour_proj = jnp.asarray(rng.normal(size=(P, 3)) / np.sqrt(P) / float(scale.mean()), jnp.float32)
    # static resource capacity map K(x) and noise-amplitude map A(x)
    gx, gy = np.meshgrid((np.arange(G) + .5) * cs, (np.arange(G) + .5) * cs, indexing="ij")
    prng = np.random.default_rng(7)
    Kmap = np.full((G, G), 0.03)
    for c in prng.uniform(0, L, size=(cfg.n_patches, 2)):
        dx = gx - c[0]; dy = gy - c[1]; dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        Kmap += np.exp(-(dx**2 + dy**2) / (2 * cfg.patch_sigma**2 * (L / 0.5) ** 2))
    Kmap = jnp.asarray(np.minimum(Kmap, 1.2), jnp.float32)
    Amap = 0.3 + Kmap / Kmap.max()
    offs = jnp.asarray([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], jnp.int32)

    def blur(f):
        return (f + jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0) + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1)) / 5.0

    def lap(f):
        return jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0) + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1) - 4 * f

    def grad(f, cell):
        gxf = (jnp.roll(f, -1, 0) - jnp.roll(f, 1, 0))[cell[:, 0], cell[:, 1]]
        gyf = (jnp.roll(f, -1, 1) - jnp.roll(f, 1, 1))[cell[:, 0], cell[:, 1]]
        return gxf, gyf

    def new_genomes(key, n):
        k1, k2 = jax.random.split(key)
        W = jax.random.normal(k1, (n, P)) * scale
        g = jax.random.uniform(k2, (n,), minval=cfg.g_lo, maxval=cfg.g_hi)
        mult = jnp.ones((n, P)).at[:, hh_off:hh_off + H * H].set(g[:, None])
        return W * mult

    def unit(key, n):
        v = jax.random.normal(key, (n, H)); return v / jnp.linalg.norm(v, axis=1, keepdims=True)

    def init(key, prm):
        ks = jax.random.split(key, 5)
        n0 = int(N * cfg.n_init_frac)
        alive = jnp.arange(N) < n0
        uid = jnp.where(alive, jnp.arange(N), -1).astype(jnp.int32)
        z = jnp.zeros(N); zi = jnp.zeros(N, jnp.int32)
        pos = jax.random.uniform(ks[0], (N, 2)) * L
        return dict(pos=pos, vel=jnp.zeros((N, 2)), E=jnp.where(alive, cfg.E0, 0.0), h=jnp.zeros((N, H)),
                    v_tan=unit(ks[1], N), lam_sum=z, lam_n=z, gain_sum=z,
                    W=new_genomes(ks[2], N), alive=alive, uid=uid, parent=jnp.full((N,), -1, jnp.int32), root=uid,
                    birth=zi, n_off=zi, prate=jnp.full((N,), jnp.nan), act=jnp.zeros((N, A_DIM)),
                    xprev=jnp.zeros((N, I_DIM)), aprev=jnp.zeros((N, A_DIM)),
                    Wp=jnp.zeros((N, H + 1, C_EXT)), ev=jnp.ones((N, C_EXT)), surp=z, mse=z,
                    xm=jnp.zeros((N, C_EXT)), xv=jnp.full((N, C_EXT), 1e-3), spd=z, anchor=pos, res=z, own=z, intake=z,
                    next_uid=jnp.int32(n0), gpar=jnp.full((N,), -1, jnp.int32), anc=uid, l0s=z, l0n=z, lstep=z, cstep=z,
                    c_int=z, c_cost=z, c_act=z, c_eat=z, c_build=z, Rf=Kmap, F=jnp.zeros((G, G)), Wl=jnp.zeros((G, G)), Pf=jnp.zeros((G, G)),
                    Fvar=jnp.zeros((G, G)), t=jnp.int32(0), key=ks[3], overflow=jnp.int32(0), rescued=jnp.int32(0))

    def colour(W):
        return jax.nn.sigmoid(2.5 * (W @ colour_proj))

    def neighbours(pos, alive):
        cc = jnp.clip((pos / L * C).astype(jnp.int32), 0, C - 1)
        cid = jnp.where(alive, cc[:, 0] * C + cc[:, 1], C * C)
        order = jnp.argsort(cid); sc = cid[order]
        start = jnp.searchsorted(sc, jnp.arange(C * C + 1))
        rank = jnp.arange(N) - start[sc]
        table = jnp.full((C * C + 1, M), N, jnp.int32).at[sc, rank].set(order.astype(jnp.int32), mode="drop")
        ov = jnp.sum((rank >= M) & (sc < C * C))
        nc = ((cc[:, None, 0] + offs[None, :, 0]) % C) * C + (cc[:, None, 1] + offs[None, :, 1]) % C
        cand = table[nc].reshape(N, 9 * M)
        return cand, ov

    def step(s, prm):
        key, kmut, kpar, ksp, kspp, kdon, knz, kv, kr = jax.random.split(s["key"], 9)
        pos, vel, E, alive = s["pos"], s["vel"], s["E"], s["alive"]
        Rf, F, Wl, Pf = s["Rf"], s["F"], s["Wl"], s["Pf"]
        fb, be = prm["feedback"], prm["build_eff"]
        age = (s["t"] - s["birth"]).astype(jnp.float32)
        # ---- neighbours (cell list)
        cand, ov = neighbours(pos, alive)
        pos_x = jnp.concatenate([pos, jnp.zeros((1, 2))]); alive_x = jnp.concatenate([alive, jnp.zeros(1, bool)])
        d = pos_x[cand] - pos[:, None, :]; d = d - L * jnp.round(d / L)
        dist = jnp.sqrt((d**2).sum(-1) + 1e-12)
        nb = alive_x[cand] & alive[:, None] & (cand != jnp.arange(N)[:, None]) & (dist < cfg.R)
        nbf = nb.astype(jnp.float32); n_nb = nbf.sum(1); inv_n = 1.0 / jnp.maximum(n_nb, 1)
        u = d / dist[..., None]
        # ---- senses
        cell = jnp.clip((pos / cs).astype(jnp.int32), 0, G - 1)
        R_here = Rf[cell[:, 0], cell[:, 1]]; F_here = F[cell[:, 0], cell[:, 1]]
        W_here = Wl[cell[:, 0], cell[:, 1]]; P_here = Pf[cell[:, 0], cell[:, 1]]
        gRx, gRy = grad(Rf, cell); gPx, gPy = grad(Pf, cell); gFx, gFy = grad(F, cell)
        sig_x = jnp.concatenate([s["act"][:, 5], jnp.zeros(1)])
        col = colour(s["W"]) - 0.5; col_x = jnp.concatenate([col, jnp.zeros((1, 3))])
        mdir = (nbf[..., None] * u).sum(1) * inv_n[:, None]
        msig = (nbf * sig_x[cand]).sum(1) * inv_n
        kin = (nbf * (col_x[cand] * col[:, None, :]).sum(-1)).sum(1) * inv_n * 8
        x = jnp.stack([R_here + cfg.noise_sense * F_here, 10 * gRx, 10 * gRy, F_here, W_here, P_here, 10 * gPx, 10 * gPy,
                       jnp.log1p(E) - 0.7, n_nb / 8, mdir[:, 0], mdir[:, 1], msig, kin, jnp.ones(N)], 1)
        xe = x[:, EXT]
        # ---- observer: online next-input predictor from own hidden state (NLMS) -> surprisal
        z = jnp.concatenate([s["h"], jnp.ones((N, 1))], 1)
        e = xe - jnp.einsum("nk,nkc->nc", z, s["Wp"])
        sup = 0.5 * jnp.sum(e**2 / s["ev"] + jnp.log(2 * jnp.pi * s["ev"]), 1)
        Wp = s["Wp"] + cfg.pred_lr * z[:, :, None] * e[:, None, :] / (1.0 + (z**2).sum(1))[:, None, None]
        ev = 0.95 * s["ev"] + 0.05 * (e**2 + 1e-4)
        surp = jnp.where(age > 0, 0.97 * s["surp"] + 0.03 * sup, sup)
        mse = 0.97 * s["mse"] + 0.03 * (e**2).mean(1)
        xm = 0.97 * s["xm"] + 0.03 * xe
        xv = 0.97 * s["xv"] + 0.03 * (xe - s["xm"])**2
        # ---- brain + tangent dynamics (jax.jvp on the recurrent step)
        g = unpack(s["W"], lay)
        drive = jnp.einsum("ni,nih->nh", x, g["W_ih"]) + g["b_h"]
        rnn = lambda hh: jnp.tanh(drive + jnp.einsum("nk,nkh->nh", hh, g["W_hh"]))
        h, jv = jax.jvp(rnn, (s["h"],), (s["v_tan"],))
        nrm = jnp.linalg.norm(jv, axis=1) + 1e-30
        v_tan = jv / nrm[:, None]
        ok = alive & (age >= cfg.lyap_burn)
        lstep = jnp.where(alive, jnp.log(nrm), 0.0)
        l0s = s["l0s"] + lstep; l0n = s["l0n"] + alive
        lam_sum = s["lam_sum"] + jnp.where(ok, jnp.log(nrm), 0.0)
        lam_n = s["lam_n"] + ok
        dsq = (1 - h**2)**2; colsq = (g["W_hh"]**2).sum(1)
        gain_sum = s["gain_sum"] + jnp.where(ok, (dsq * colsq).sum(1) / H, 0.0)
        a = jnp.tanh(jnp.einsum("nh,nha->na", h, g["W_ho"]) + g["b_o"])
        thrust, eat, rep = a[:, :2], (a[:, 2] + 1) / 2, a[:, 3]
        share, attack = jnp.maximum(a[:, 4], 0), jnp.maximum(-a[:, 4], 0)
        signal, bond, build = a[:, 5], a[:, 6], (a[:, 7] + 1) / 2
        # ---- motion
        overlap = jnp.maximum(cfg.r0 - dist, 0) * nbf
        f_rep = -(overlap[..., None] * u).sum(1) * cfg.k_rep / cfg.r0
        bpos = jnp.concatenate([jnp.maximum(bond, 0), jnp.zeros(1)])
        bb = jnp.minimum(bpos[:N, None], bpos[cand]) * nbf
        f_adh = ((bb * (dist - 1.5 * cfg.r0))[..., None] * u).sum(1) * cfg.k_adh
        push = -cfg.k_push * jnp.stack([gFx, gFy], 1)
        drag = 1.0 / (1.0 + cfg.wall_drag * W_here * be)
        vel = (cfg.damp * vel + cfg.a_max * thrust + 0.01 * (f_rep + f_adh) + push * cfg.a_max) * drag[:, None]
        vel = jnp.where(alive[:, None], vel, 0.0)
        pos = (pos + vel) % L
        # ---- harvest (demand-limited; variable where F is large)
        flat = cell[:, 0] * G + cell[:, 1]
        want = jnp.where(alive, eat * cfg.harvest * R_here * jnp.clip(1 + 2 * F_here, 0, 2), 0.0)
        demand = jnp.zeros(G * G).at[flat].add(want)
        frac = jnp.minimum(1.0, Rf.reshape(-1) / jnp.maximum(demand, 1e-9))
        got = want * frac[flat]
        # ---- social flows over neighbour lists
        give = share * cfg.share_rate * E * (n_nb > 0)
        recv = jnp.zeros(N + 1).at[cand.reshape(-1)].add((nbf * (give * inv_n)[:, None]).reshape(-1))[:N]
        E_x = jnp.concatenate([E, jnp.zeros(1)])
        stolen_pair = nbf * (attack * cfg.attack_rate * inv_n)[:, None] * E_x[cand]
        lost = jnp.zeros(N + 1).at[cand.reshape(-1)].add(stolen_pair.reshape(-1))[:N]
        dE_soc = -give + recv - lost + cfg.attack_eff * stolen_pair.sum(1)
        # ---- costs (+ toxicity of P)
        cost = (prm["c_basal"] + cfg.c_move * (thrust**2).sum(1) + cfg.c_eat * eat + cfg.c_sig * jnp.abs(signal)
                + cfg.c_bond * jnp.maximum(bond, 0) + cfg.c_att * attack + cfg.c_build * build
                + fb * cfg.c_tox * P_here**2) * prm["cost_mul"]
        E = jnp.where(alive, jnp.minimum(E + got + dE_soc - cost, cfg.E_cap), 0.0)
        intake = 0.97 * s["intake"] + 0.03 * got
        cstep = jnp.where(alive, cost, 0.0)
        c_cost = s["c_cost"] + cstep; c_int = s["c_int"] + jnp.where(alive, got, 0.0)
        c_act = s["c_act"] + jnp.where(alive, (thrust**2).sum(1), 0.0)
        c_eat = s["c_eat"] + jnp.where(alive, eat, 0.0); c_build = s["c_build"] + jnp.where(alive, build, 0.0)
        alive_before = alive
        alive = alive & (E > 0)
        # ---- fields
        got_map = jnp.zeros(G * G).at[flat].add(got).reshape(G, G)
        R_dyn = Rf - got_map
        R_dyn = R_dyn + cfg.regrow * R_dyn * (1 - R_dyn / Kmap) * (1 + cfg.fert * Pf) + cfg.seed_regrow * (Kmap - R_dyn)
        Rf = fb * jnp.clip(R_dyn, 0, 1.5) + (1 - fb) * Kmap
        dep_w = jnp.zeros(G * G).at[flat].add(jnp.where(alive_before, build * cfg.wall_dep, 0.0)).reshape(G, G)
        Wl = Wl * (1 - cfg.wall_decay) + dep_w
        dep_p = jnp.zeros(G * G).at[flat].add(jnp.where(alive_before, cfg.waste_frac * cost + cfg.p_dep_sig * jnp.abs(signal), 0.0)).reshape(G, G)
        Pf = jnp.maximum(Pf + cfg.p_diff * lap(Pf) - cfg.p_decay * Pf + dep_p, 0)
        Weff = Wl * be
        th = cfg.theta + cfg.k_theta * Weff / (1.0 + Weff)          # saturating: keeps OU stable (th <= theta + k_theta)
        amp = prm["env_var"] * Amap * jnp.exp(-cfg.k_wall * Weff)
        F = F * (1 - th) + amp * jnp.sqrt(2 * cfg.theta) * blur(jax.random.normal(knz, (G, G))) * jnp.sqrt(5.0)
        Fvar = 0.98 * s["Fvar"] + 0.02 * F**2
        # ---- reproduction into free slots
        want_r = alive & (rep > 0) & (E > cfg.E_rep)
        par = jnp.nonzero(want_r, size=N, fill_value=0)[0]
        slots = jnp.nonzero(~alive, size=N, fill_value=0)[0]
        n = jnp.minimum(want_r.sum(), (~alive).sum())
        k = jnp.arange(N); valid = k < n
        donor_s = jax.random.choice(kdon, N, (N,), p=alive / jnp.maximum(alive.sum(), 1))
        donor = jnp.where(prm["shuffle"] > 0.5, donor_s, par)
        Wg = s["W"]
        child_W = Wg[donor] + prm["mut_sigma"] * scale * jax.random.normal(kmut, (N, P))
        s_eff = jnp.where(valid, slots, N); p_eff = jnp.where(valid, par, N)
        page = (s["t"] - s["birth"][par]).astype(jnp.float32) + 1
        prate_child = s["n_off"][par] / page
        child_E = 0.4 * E[par]
        E = E.at[p_eff].add(-0.5 * E[par], mode="drop")
        n_off = s["n_off"].at[p_eff].add(1, mode="drop")
        newuid = s["next_uid"] + k.astype(jnp.int32)
        cpos = (pos[par] + 0.004 * jax.random.normal(kpar, (N, 2))) % L
        def put(arr, val):
            return arr.at[s_eff].set(val, mode="drop")
        st = dict(W=put(Wg, child_W), E=put(E, child_E), pos=put(pos, cpos), vel=put(vel, 0.0), h=put(h, 0.0),
                  v_tan=put(v_tan, unit(kv, N)), lam_sum=put(lam_sum, 0.0), lam_n=put(lam_n, 0.0), gain_sum=put(gain_sum, 0.0),
                  alive=put(alive, True), uid=put(s["uid"], newuid), parent=put(s["parent"], s["uid"][par]),
                  root=put(s["root"], s["root"][donor]), birth=put(s["birth"], s["t"]), n_off=put(n_off, 0),
                  prate=put(s["prate"], prate_child), Wp=put(Wp, 0.0), ev=put(ev, 1.0), surp=put(surp, 0.0), mse=put(mse, 0.0),
                  xm=put(xm, 0.0), xv=put(xv, 1e-3), spd=put(s["spd"], 0.0), anchor=put(s["anchor"], cpos), res=put(s["res"], 0.0),
                  own=put(s["own"], 0.0), intake=put(intake, 0.0),
                  gpar=put(s["gpar"], s["uid"][donor]), anc=put(s["anc"], s["anc"][donor]), l0s=put(l0s, 0.0), l0n=put(l0n, 0.0),
                  lstep=put(lstep, 0.0), cstep=put(cstep, 0.0), c_int=put(c_int, 0.0), c_cost=put(c_cost, 0.0),
                  c_act=put(c_act, 0.0), c_eat=put(c_eat, 0.0), c_build=put(c_build, 0.0))
        next_uid = s["next_uid"] + n.astype(jnp.int32)
        # ---- spawns: background trickle + rescue when nearly extinct
        SP = cfg.rescue_per_step
        pop = st["alive"].sum()
        n_sp = jnp.where(pop < cfg.min_pop_frac * N, SP, (jax.random.uniform(kspp) < cfg.spawn_prob).astype(jnp.int32))
        free = jnp.nonzero(~st["alive"], size=SP, fill_value=0)[0]
        okk = (jnp.arange(SP) < jnp.minimum(n_sp, (~st["alive"]).sum()))
        f_eff = jnp.where(okk, free, N)
        ks1, ks2, ks3 = jax.random.split(ksp, 3)
        spos = jax.random.uniform(ks2, (SP, 2)) * L
        su = next_uid + jnp.arange(SP, dtype=jnp.int32)
        def sput(arr, val):
            return arr.at[f_eff].set(val, mode="drop")
        st.update(W=sput(st["W"], new_genomes(ks1, SP)), E=sput(st["E"], cfg.E0), pos=sput(st["pos"], spos),
                  vel=sput(st["vel"], 0.0), h=sput(st["h"], 0.0), v_tan=sput(st["v_tan"], unit(ks3, SP)),
                  lam_sum=sput(st["lam_sum"], 0.0), lam_n=sput(st["lam_n"], 0.0), gain_sum=sput(st["gain_sum"], 0.0),
                  alive=sput(st["alive"], True), uid=sput(st["uid"], su), root=sput(st["root"], su), parent=sput(st["parent"], -1),
                  birth=sput(st["birth"], s["t"]), n_off=sput(st["n_off"], 0), prate=sput(st["prate"], jnp.nan),
                  Wp=sput(st["Wp"], 0.0), ev=sput(st["ev"], 1.0), surp=sput(st["surp"], 0.0), mse=sput(st["mse"], 0.0),
                  xm=sput(st["xm"], 0.0), xv=sput(st["xv"], 1e-3), spd=sput(st["spd"], 0.0), anchor=sput(st["anchor"], spos),
                  res=sput(st["res"], 0.0), own=sput(st["own"], 0.0), intake=sput(st["intake"], 0.0),
                  gpar=sput(st["gpar"], -1), anc=sput(st["anc"], -1),
                  **{k: sput(st[k], 0.0) for k in ("l0s", "l0n", "lstep", "cstep", "c_int", "c_cost", "c_act", "c_eat", "c_build")})
        next_uid = next_uid + okk.sum().astype(jnp.int32)
        rescued = s["rescued"] + jnp.where(pop < cfg.min_pop_frac * N, okk.sum(), 0).astype(jnp.int32)
        # ---- behavioural observables: speed, residence, own-wall deposit near current position
        sp_now = jnp.linalg.norm(vel, axis=1)
        dd = st["pos"] - st["anchor"]; dd = dd - L * jnp.round(dd / L)
        moved = jnp.sqrt((dd**2).sum(1)) > cfg.R
        anchor = jnp.where(moved[:, None], st["pos"], st["anchor"])
        res = jnp.where(moved, 0.0, st["res"] + 1)
        spd = 0.95 * st["spd"] + 0.05 * sp_now
        own = st["own"] * (1 - cfg.wall_decay) * jnp.exp(-sp_now / cs) + jnp.where(alive_before, build * cfg.wall_dep, 0.0)
        act = jnp.where(st["alive"][:, None], a, 0.0)
        st.update(anchor=anchor, res=res, spd=spd, own=jnp.where(st["alive"], own, 0.0), act=act, xprev=x, aprev=s["act"],
                  next_uid=next_uid, Rf=Rf, F=F, Wl=Wl, Pf=Pf, Fvar=Fvar, t=s["t"] + 1, key=key,
                  overflow=s["overflow"] + ov.astype(jnp.int32), rescued=rescued)
        return st

    def chunk_one(s, prm, n_steps):
        def body(c, _):
            return step(c, prm), None
        s, _ = jax.lax.scan(body, s, None, length=n_steps)
        return s

    consts = dict(Kmap=np.asarray(Kmap), Amap=np.asarray(Amap), hh_off=hh_off, P=P, C=C, lay=lay, step=step)
    return init, chunk_one, colour, consts

def batched(cfg: Cfg):
    init, chunk_one, colour, consts = make(cfg)
    binit = jax.jit(jax.vmap(init))
    cache = {}
    def bchunk(states, prms, n_steps):
        if n_steps not in cache:
            cache[n_steps] = jax.jit(jax.vmap(lambda s, p: chunk_one(s, p, n_steps)))
        return cache[n_steps](states, prms)
    bcolour = jax.jit(jax.vmap(colour))
    return binit, bchunk, bcolour, consts

def stack_params(plist):
    return {k: jnp.stack([p[k] for p in plist]) for k in PARAM_KEYS}
