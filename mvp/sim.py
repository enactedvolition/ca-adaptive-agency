"""Tiny thermodynamic agent-CA in JAX: random Elman-RNN particles with an
energy budget. Fraser-Price-*style* (his code is not public; nothing here is
his implementation). Built to exercise the viz architecture, not to be fast.

Every slot is an agent with its own genome (flat RNN weights). Senses ->
RNN -> actions: thrust(2), eat, reproduce, share(+)/attack(-), signal, bond.
Everything costs energy; E<=0 kills. No training: random spawns + mutation.

Variants (the honesty layer's controls), same seed:
  full         - as described
  no_cost      - actions and existence are free (no selection pressure)
  shuffled     - children inherit a random living agent's genome, not the parent's
  memoryless   - hidden state zeroed every step (RNN -> feed-forward)

Output: zarr store of strided snapshots + provenance.json per run.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from dataclasses import dataclass, asdict, replace
import numpy as np
import jax, jax.numpy as jnp
import zarr

@dataclass(frozen=True)
class Cfg:
    N: int = 1536          # agent slots (fixed-size arrays, alive mask)
    H: int = 16            # RNN hidden size
    G: int = 96            # energy-field grid
    steps: int = 2500
    snap_every: int = 10
    seed: int = 0
    variant: str = "full"
    n_init: int = 400
    R: float = 0.03        # sensing / interaction radius (world is unit torus)
    r0: float = 0.008      # hard-core radius
    a_max: float = 0.002
    damp: float = 0.8
    k_rep: float = 0.3
    k_adh: float = 0.15
    E0: float = 1.0
    E_rep: float = 1.6
    E_cap: float = 4.0
    c_basal: float = 0.004
    c_move: float = 0.01
    c_eat: float = 0.002
    c_sig: float = 0.001
    c_bond: float = 0.001
    c_att: float = 0.004
    harvest: float = 0.12
    share_rate: float = 0.05
    attack_rate: float = 0.08
    attack_eff: float = 0.5
    regen: float = 0.02
    n_blobs: int = 5
    blob_sigma: float = 0.07
    mut_sigma: float = 0.08
    spawn_per_step: int = 2

I_DIM, A_DIM = 10, 7

def genome_layout(H):
    shapes = [("W_ih", (I_DIM, H)), ("W_hh", (H, H)), ("b_h", (H,)), ("W_ho", (H, A_DIM)), ("b_o", (A_DIM,))]
    scales = {"W_ih": 1.5 / np.sqrt(I_DIM), "W_hh": 1.2 / np.sqrt(H), "b_h": 0.1, "W_ho": 1.5 / np.sqrt(H), "b_o": 0.3}
    off, lay, sc = 0, [], []
    for n, s in shapes:
        sz = int(np.prod(s)); lay.append((n, s, off, sz)); sc.append(np.full(sz, scales[n])); off += sz
    return lay, off, jnp.asarray(np.concatenate(sc), jnp.float32)

def unpack(W, lay):
    return {n: W[:, o:o + sz].reshape((W.shape[0],) + s) for n, s, o, sz in lay}

def make(cfg: Cfg):
    lay, P, scale = genome_layout(cfg.H)
    N, G = cfg.N, cfg.G
    rng = np.random.default_rng(1234)                 # colour projection is fixed across runs
    colour_proj = jnp.asarray(rng.normal(size=(P, 3)) / np.sqrt(P) / scale.mean(), jnp.float32)
    gx, gy = np.meshgrid((np.arange(G) + .5) / G, (np.arange(G) + .5) / G, indexing="ij")
    gx, gy = jnp.asarray(gx), jnp.asarray(gy)
    v = cfg.variant

    def source(t):
        k = jnp.arange(cfg.n_blobs)
        ang = 2 * jnp.pi * k / cfg.n_blobs + t * 0.0004 * (1 + 0.3 * k)
        cx = 0.5 + 0.3 * jnp.cos(ang); cy = 0.5 + 0.3 * jnp.sin(ang * 1.3)
        dx = gx[None] - cx[:, None, None]; dy = gy[None] - cy[:, None, None]
        dx = dx - jnp.round(dx); dy = dy - jnp.round(dy)
        return 0.02 + jnp.exp(-(dx**2 + dy**2) / (2 * cfg.blob_sigma**2)).sum(0)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        alive = jnp.arange(N) < cfg.n_init
        uid = jnp.where(alive, jnp.arange(N), -1).astype(jnp.int32)
        return dict(pos=jax.random.uniform(k1, (N, 2)), vel=jnp.zeros((N, 2)),
                    E=jnp.where(alive, cfg.E0, 0.0), h=jnp.zeros((N, cfg.H)),
                    W=jax.random.normal(k2, (N, P)) * scale, alive=alive, uid=uid,
                    parent=jnp.full((N,), -1, jnp.int32), root=uid, birth=jnp.zeros((N,), jnp.int32),
                    act=jnp.zeros((N, A_DIM)), next_uid=jnp.int32(cfg.n_init),
                    field=source(0.0), t=jnp.int32(0), key=k3)

    def colour(W):
        return jax.nn.sigmoid(2.5 * (W @ colour_proj))

    def step(s, _):
        key, kmut, kpar, ksp1, ksp2, ksp3, kdon = jax.random.split(s["key"], 7)
        pos, vel, E, alive, field = s["pos"], s["vel"], s["E"], s["alive"], s["field"]
        # ---- geometry
        d = pos[None, :, :] - pos[:, None, :]; d = d - jnp.round(d)           # d[i,j] = j - i
        dist = jnp.sqrt((d**2).sum(-1) + 1e-12)
        nb = alive[:, None] & alive[None, :] & (dist < cfg.R) & ~jnp.eye(N, dtype=bool)
        nbf = nb.astype(jnp.float32); n_nb = nbf.sum(1)
        # ---- senses
        cell = jnp.clip((pos * G).astype(jnp.int32), 0, G - 1)
        f_here = field[cell[:, 0], cell[:, 1]]
        fgx = (jnp.roll(field, -1, 0) - jnp.roll(field, 1, 0))[cell[:, 0], cell[:, 1]]
        fgy = (jnp.roll(field, -1, 1) - jnp.roll(field, 1, 1))[cell[:, 0], cell[:, 1]]
        unit = d / dist[..., None]
        mdir = (nbf[..., None] * unit).sum(1) / jnp.maximum(n_nb, 1)[:, None]
        sig = s["act"][:, 5]
        msig = (nbf * sig[None, :]).sum(1) / jnp.maximum(n_nb, 1)
        col = colour(s["W"]); coln = col - 0.5
        kin = (nbf * (coln @ coln.T)).sum(1) / jnp.maximum(n_nb, 1) * 8
        x = jnp.stack([f_here, 10 * fgx, 10 * fgy, jnp.log1p(E) - 0.7, n_nb / 8,
                       mdir[:, 0], mdir[:, 1], msig, kin, jnp.ones(N)], 1)
        # ---- brain
        g = unpack(s["W"], lay)
        h = s["h"] * (0.0 if v == "memoryless" else 1.0)
        h = jnp.tanh(jnp.einsum("ni,nih->nh", x, g["W_ih"]) + jnp.einsum("nk,nkh->nh", h, g["W_hh"]) + g["b_h"])
        a = jnp.tanh(jnp.einsum("nh,nha->na", h, g["W_ho"]) + g["b_o"])
        thrust, eat, rep = a[:, :2], (a[:, 2] + 1) / 2, a[:, 3]
        share, attack = jnp.maximum(a[:, 4], 0), jnp.maximum(-a[:, 4], 0)
        signal, bond = a[:, 5], a[:, 6]
        # ---- forces & motion
        overlap = jnp.maximum(cfg.r0 - dist, 0) * nbf
        f_rep = -(overlap[..., None] * unit).sum(1) * cfg.k_rep / cfg.r0
        bb = jnp.minimum(jnp.maximum(bond, 0)[:, None], jnp.maximum(bond, 0)[None, :]) * nbf
        f_adh = ((bb * (dist - 1.5 * cfg.r0))[..., None] * unit).sum(1) * cfg.k_adh
        vel = cfg.damp * vel + cfg.a_max * thrust + 0.01 * (f_rep + f_adh)
        vel = jnp.where(alive[:, None], vel, 0.0)
        pos = (pos + vel) % 1.0
        # ---- eating (demand-limited per cell)
        flat = cell[:, 0] * G + cell[:, 1]
        want = jnp.where(alive, eat * cfg.harvest * f_here, 0.0)
        demand = jnp.zeros(G * G).at[flat].add(want)
        frac = jnp.minimum(1.0, field.reshape(-1) / jnp.maximum(demand, 1e-9))
        got = want * frac[flat]
        field = jnp.maximum(field.reshape(-1).at[flat].add(-got), 0).reshape(G, G)
        field = field + cfg.regen * (source(s["t"].astype(jnp.float32)) - field)
        # ---- share / attack flows  F[i,j]: energy leaving i towards j
        F = nbf * (share * cfg.share_rate * E / jnp.maximum(n_nb, 1))[:, None]
        steal = nbf * (attack * cfg.attack_rate / jnp.maximum(n_nb, 1))[:, None]   # i attacks j
        S = steal.T * E[:, None]                                                  # S[i,j]: j steals from i
        dE_social = -F.sum(1) + F.sum(0) - S.sum(1) + cfg.attack_eff * S.sum(0)
        # ---- costs
        cost = (cfg.c_basal + cfg.c_move * (thrust**2).sum(1) + cfg.c_eat * eat
                + cfg.c_sig * jnp.abs(signal) + cfg.c_bond * jnp.maximum(bond, 0) + cfg.c_att * attack)
        if v == "no_cost":
            cost = cost * 0.0
        E = jnp.where(alive, jnp.minimum(E + got + dE_social - cost, cfg.E_cap), 0.0)
        alive = alive & (E > 0)
        # ---- reproduction into free slots
        want_r = alive & (rep > 0) & (E > cfg.E_rep)
        par = jnp.nonzero(want_r, size=N, fill_value=0)[0]
        slots = jnp.nonzero(~alive, size=N, fill_value=0)[0]
        n = jnp.minimum(want_r.sum(), (~alive).sum())
        k = jnp.arange(N); valid = k < n
        if v == "shuffled":
            donor = jax.random.choice(kdon, N, (N,), p=alive / jnp.maximum(alive.sum(), 1))
        else:
            donor = par
        W = s["W"]
        child_W = W[donor] + cfg.mut_sigma * scale * jax.random.normal(kmut, (N, P))
        s_eff = jnp.where(valid, slots, N); p_eff = jnp.where(valid, par, N)
        child_E = 0.4 * E[par]
        E = E.at[p_eff].add(-0.5 * E[par], mode="drop")
        newuid = s["next_uid"] + k.astype(jnp.int32)
        W = W.at[s_eff].set(child_W, mode="drop")
        E = E.at[s_eff].set(child_E, mode="drop")
        pos = pos.at[s_eff].set((pos[par] + 0.004 * jax.random.normal(kpar, (N, 2))) % 1.0, mode="drop")
        vel = vel.at[s_eff].set(0.0, mode="drop"); h = h.at[s_eff].set(0.0, mode="drop")
        alive = alive.at[s_eff].set(True, mode="drop")
        uid = s["uid"].at[s_eff].set(newuid, mode="drop")
        parent = s["parent"].at[s_eff].set(s["uid"][par], mode="drop")
        root = s["root"].at[s_eff].set(s["root"][donor], mode="drop")
        birth = s["birth"].at[s_eff].set(s["t"], mode="drop")
        next_uid = s["next_uid"] + n.astype(jnp.int32)
        # ---- random spawns (abiogenesis)
        SP = cfg.spawn_per_step
        free = jnp.nonzero(~alive, size=SP, fill_value=0)[0]
        ok = (jnp.arange(SP) < (~alive).sum())
        f_eff = jnp.where(ok, free, N)
        W = W.at[f_eff].set(jax.random.normal(ksp1, (SP, P)) * scale, mode="drop")
        E = E.at[f_eff].set(cfg.E0, mode="drop")
        pos = pos.at[f_eff].set(jax.random.uniform(ksp2, (SP, 2)), mode="drop")
        vel = vel.at[f_eff].set(0.0, mode="drop"); h = h.at[f_eff].set(0.0, mode="drop")
        alive = alive.at[f_eff].set(True, mode="drop")
        su = next_uid + jnp.arange(SP, dtype=jnp.int32)
        uid = uid.at[f_eff].set(su, mode="drop"); root = root.at[f_eff].set(su, mode="drop")
        parent = parent.at[f_eff].set(-1, mode="drop"); birth = birth.at[f_eff].set(s["t"], mode="drop")
        next_uid = next_uid + ok.sum().astype(jnp.int32)
        act = jnp.where(alive[:, None], a, 0.0)
        return dict(pos=pos, vel=vel, E=E, h=h, W=W, alive=alive, uid=uid, parent=parent, root=root,
                    birth=birth, act=act, next_uid=next_uid, field=field, t=s["t"] + 1, key=key), None

    @jax.jit
    def chunk(s):
        s, _ = jax.lax.scan(step, s, None, length=cfg.snap_every)
        return s

    return init, chunk, colour, P

def run(cfg: Cfg, out_dir: str):
    init, chunk, colour, P = make(cfg)
    T = cfg.steps // cfg.snap_every
    os.makedirs(out_dir, exist_ok=True)
    store = zarr.open_group(os.path.join(out_dir, "snapshots.zarr"), mode="w")
    N, H, G = cfg.N, cfg.H, cfg.G
    spec = {"pos": ((T, N, 2), "f4"), "E": ((T, N), "f2"), "alive": ((T, N), "bool"),
            "uid": ((T, N), "i4"), "parent": ((T, N), "i4"), "root": ((T, N), "i4"),
            "birth": ((T, N), "i4"), "colour": ((T, N, 3), "f2"), "h": ((T, N, H), "f2"),
            "act": ((T, N, A_DIM), "f2"), "field": ((T, G, G), "f2")}
    arr = {k: store.create_array(k, shape=sh, chunks=(1,) + sh[1:], dtype=dt) for k, (sh, dt) in spec.items()}
    s = init(jax.random.PRNGKey(cfg.seed))
    t0 = time.time()
    for i in range(T):
        s = chunk(s)
        snap = {k: np.asarray(s[k]) for k in ("pos", "E", "alive", "uid", "parent", "root", "birth", "h", "act", "field")}
        snap["colour"] = np.asarray(colour(s["W"]))
        for k in spec:
            arr[k][i] = snap[k]
        if i % 50 == 0:
            print(f"[{cfg.variant}] t={int(s['t'])} pop={int(snap['alive'].sum())} meanE={snap['E'][snap['alive']].mean():.2f} {time.time()-t0:.0f}s", flush=True)
    cfg_json = json.dumps(asdict(cfg), sort_keys=True)
    prov = {"schema": "ca-agency-run/0", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": asdict(cfg), "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
            "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "genome_params": int(P), "inputs": I_DIM, "actions": A_DIM,
            "env": {"python": sys.version.split()[0], "jax": jax.__version__, "numpy": np.__version__,
                    "zarr": zarr.__version__, "platform": platform.platform(), "backend": jax.default_backend()},
            "snapshot_caveat": "strided: agents born and dead between snapshots are absent from the phylogeny",
            "wall_seconds": round(time.time() - t0, 1)}
    json.dump(prov, open(os.path.join(out_dir, "provenance.json"), "w"), indent=2)
    return prov

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    cfg = replace(Cfg(), variant=a.variant, steps=a.steps, seed=a.seed)
    prov = run(cfg, os.path.join(a.out, a.variant))
    print(f"done {a.variant}: {prov['wall_seconds']}s")
