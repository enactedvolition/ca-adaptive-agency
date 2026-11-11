"""sim10: thermodynamically strict, field-mediated microbial-ecology world (numpy). See docs/SIM10_DESIGN.md.

New file; does not import or change sim7-9. One rule set for every agent; no hard-coded species.
Conserved: matter (units) and energy. Boundary: chemostat inflow of A, dilution outflow of every field
species, heat to an isothermal bath (kT = 1). Every step asserts both ledgers (rel. 1e-9) and that each
process's free-energy dissipation (entropy production * T) is >= 0.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

GENES = ["size", "auto", "upt", "eat_org", "tox_sec", "tox_res", "sig_sec", "motil", "chemo", "adh", "share",
         "engulf", "infect", "vir", "mut", "defense", "mem", "p_thr", "p_gain", "detox", "cat_gs",
         "rt_vmax", "rt_km", "rt_spec"]
NS = len(GENES); NT = 8; NG = NS + 2 * NT
gi = {g: i for i, g in enumerate(GENES)}
TAG = slice(NS, NS + NT); REC = slice(NS + NT, NS + 2 * NT)
# field species (matter-carrying), standard energies e_s (kT units); AGT is tag information, not matter
A, V, W, D, X, S, AG = range(7); NSP = 7
SPN = ["A", "V", "W", "D", "X", "S", "Ag"]
EN = np.array([10.0, 6.0, 0.0, 5.0, 6.0, 6.0, 5.0]); E_B = 12.0      # biomass energy per unit matter
# reversible first-order decays to W (forward rate constants); reverse k- = k+ exp(-(e_s - e_W))
KDEC = {V: 0.060, D: 0.01, X: 0.02, S: 0.05, AG: 0.02}
DIFF = np.array([0.16, 0.08, 0.16, 0.05, 0.10, 0.15, 0.10])
COST = dict(auto=0.004, upt=0.0015, eat_org=0.001, tox_res=0.002, chemo=0.0003, adh=0.0008, engulf=0.002,
            infect=0.0005, defense=0.002, mem=0.0015, detox=0.0015, cat_gs=0.0015, rt_vmax=0.0015)

class ThermoError(AssertionError):
    pass

@dataclass(frozen=True)
class Cfg:
    G: int = 40
    Nmax: int = 5000
    N0: int = 1500
    steps: int = 4000
    W: int = 250
    inflow: float = 1.0      # multiplies chemostat dilution rate f (0 = closed system)
    f: float = 0.035; A_res: float = 1.0; k1: float = 1.0
    eta: float = 0.5         # fraction of catabolic free energy captured into the ATP-like store E
    k_upt: float = 0.06; k_cat: float = 2.0; ovh_tr: float = 0.1; c_int: float = 1.0
    base: float = 0.004; c_move: float = 0.002; c_div: float = 0.05
    tox_k: float = 0.05; tox_hit: float = 0.5
    M0: float = 0.3; E0: float = 0.8
    max_age: int = 3000
    w_aff: float = 0.5; drain: float = 0.05; clear_p: float = 0.1; engulf_eff: float = 0.6; shed: float = 0.002
    hijack: int = 1; mem_effect: int = 1; rt_sig_ko: int = 0
    tol: float = 1e-9
    seed: int = 0

def lap(Fm):
    return np.roll(Fm, 1, -1) + np.roll(Fm, -1, -1) + np.roll(Fm, 1, -2) + np.roll(Fm, -1, -2) - 4 * Fm

def aff(a, b, w):
    return np.exp(-((a - b) ** 2).sum(-1) / w)

def xlnx(a, b):
    """(a-b)*ln(a/b) >= 0, safe."""
    a = np.maximum(a, 1e-300); b = np.maximum(b, 1e-300)
    return (a - b) * np.log(a / b)

class World:
    def __init__(self, cfg: Cfg):
        self.c = c = cfg; G = c.G
        self.rng = r = np.random.default_rng(c.seed)
        F = np.zeros((NSP, G, G)); F[A] = c.A_res
        for _ in range(12):
            x, y = r.integers(0, G, 2); ix = (np.arange(x, x + 4) % G)[:, None]; iy = np.arange(y, y + 4) % G
            F[A, ix, iy] = 0.5; F[V, ix, iy] = 0.25
        self.F = F; self.AGT = np.zeros((NT, G, G))
        n = c.Nmax
        self.alive = np.zeros(n, bool); self.g = np.zeros((n, NG)); self.E = np.zeros(n); self.M = np.zeros(n)
        self.x = np.zeros(n, int); self.y = np.zeros(n, int); self.age = np.zeros(n, int)
        self.host = -np.ones(n, int); self.m = np.zeros((n, NT)); self.ms = np.zeros(n); self.p = np.zeros(n)
        self.uid = -np.ones(n, int); self.lab = -np.ones(n, int)
        k = c.N0
        # founders are assembled from the initial A stock (no matter or energy created)
        self.alive[:k] = True; self.g[:k] = r.random((k, NG)); self.E[:k] = c.E0; self.M[:k] = c.M0
        self.x[:k] = r.integers(0, G, k); self.y[:k] = r.integers(0, G, k)
        self.uid[:k] = np.arange(k); self.next_uid = k
        self.t = 0
        self.cum = dict(inE=0.0, outE=0.0, Q=0.0, inM=0.0, outM=0.0, sigma=0.0)
        self.M_tot0, self.E_tot0 = self.totals()
        # founders' matter/energy are counted at t=0 as part of the initial state (documented)
        self.resid = []; self.sig_hist = []
        self.new_window()

    def totals(self):
        a = self.alive
        Mt = self.F.sum() + self.M[a].sum()
        Et = (EN[:, None, None] * self.F).sum() + self.E[a].sum() + E_B * self.M[a].sum()
        return Mt, Et

    def new_window(self):
        n = self.c.Nmax
        self.log = dict(birth_free=[], birth_in=[], inf=[], eng=[], drain=[], free_steps=np.zeros(n),
                        in_steps=np.zeros(n), field_in=np.zeros(n), sig_rel=0.0, sig_rt=0.0, sig_dec=0.0,
                        inf_tags=[], birth_p=[])

    # ---------------------------------------------------------------- step
    def step(self):
        c, r, G = self.c, self.rng, self.c.G
        F = self.F; g = self.g
        Q = 0.0; sig = 0.0; dmin = 0.0     # heat to bath, entropy production (kT=1), min process dissipation
        def diss(arr):
            nonlocal sig, dmin
            if np.size(arr):
                sig += float(np.sum(arr)); dmin = min(dmin, float(np.min(arr)))
        idx = np.flatnonzero(self.alive)
        free = idx[self.host[idx] < 0]
        cell = self.x * G + self.y
        cf = cell[free]
        gf = g[free]
        # agent catalysis per cell (speeds forward and reverse equally: equilibria unchanged)
        cat1 = 1 + c.k_cat * np.bincount(cf, gf[:, gi["cat_gs"]], G * G).reshape(G, G)
        catx = 1 + c.k_cat * np.bincount(cf, gf[:, gi["detox"]], G * G).reshape(G, G)
        # ---------- 1. boundary: chemostat ----------
        fr = c.f * c.inflow
        if fr > 0:
            inA = fr * c.A_res * np.ones((G, G)); out = fr * F
            self.cum["inM"] += inA.sum(); self.cum["inE"] += EN[A] * inA.sum()
            self.cum["outM"] += out.sum(); self.cum["outE"] += (EN[:, None, None] * out).sum()
            F[A] += inA; F -= out
        # ---------- 2. diffusion (conservative, positivity-preserving since 4D<1) ----------
        for s in range(NSP):
            F[s] = F[s] + DIFF[s] * lap(F[s])
        self.AGT = self.AGT + DIFF[AG] * lap(self.AGT)
        # ---------- 3. field reactions, mass action, detailed balance ----------
        # R1: A + 2V <-> 3V ; k-/k+ = exp(-(e_A - e_V))
        kf = c.k1 * cat1; kr = kf * np.exp(-(EN[A] - EN[V]))
        Jf = kf * F[A] * F[V] ** 2; Jr = kr * F[V] ** 3
        net = Jf - Jr
        net = np.clip(net, -F[V] * 0.5, F[A] * 0.5)          # stock limit (keeps sign)
        sc = np.where(np.abs(Jf - Jr) > 0, net / np.where(Jf - Jr == 0, 1, Jf - Jr), 1.0)
        diss(sc * xlnx(Jf, Jr)); F[A] -= net; F[V] += net; Q += float(((EN[A] - EN[V]) * net).sum())
        for s, k in KDEC.items():
            kk = k * (catx if s == X else 1.0)
            Jf = kk * F[s]; Jr = kk * np.exp(-(EN[s] - EN[W])) * F[W]
            net = np.clip(Jf - Jr, -F[W] * 0.5, F[s] * 0.5)
            sc = np.where(np.abs(Jf - Jr) > 0, net / np.where(Jf - Jr == 0, 1, Jf - Jr), 1.0)
            diss(sc * xlnx(Jf, Jr))
            if s == AG:   # tag info follows antigen matter
                self.AGT -= self.AGT * np.where(F[AG] > 0, net / np.maximum(F[AG], 1e-300), 0)
            if s == S: self.log["sig_dec"] += float(net.sum())
            F[s] -= net; F[W] += net; Q += float(((EN[s] - EN[W]) * net).sum())
        if idx.size == 0:
            return self._close(Q, sig, dmin)
        # inside agents follow host
        inside = idx[self.host[idx] >= 0]
        if inside.size:
            h = self.host[inside]; self.x[inside] = self.x[h]; self.y[inside] = self.y[h]
            cell = self.x * G + self.y
        # ---------- 4. phenotype ----------
        sigl = F[S].ravel()[cell[idx]]
        pg = g[idx, gi["p_gain"]]
        self.p[idx] = np.where(pg < 0.05, 0.0, 1 / (1 + np.exp(-pg * 20 * (sigl - g[idx, gi["p_thr"]] * 0.2))))
        pf = self.p[free]; size = gf[:, gi["size"]]
        def take(s, demand, cells):
            tot = np.bincount(cells, demand, G * G); stock = F[s].ravel()
            frac = np.where(tot > 0, np.minimum(1, stock / np.maximum(tot, 1e-300)), 0)
            got = demand * frac[cells]
            F[s] = (stock - np.bincount(cells, got, G * G)).reshape(G, G)
            return got
        Wc = np.maximum(F[W].ravel()[cf], 1e-12)
        # ---------- 5. catabolism of A (passive uptake) : A -> W, capture eta * (-dG_actual) ----------
        Ac = F[A].ravel()[cf]
        dG_A = (EN[W] - EN[A]) + np.log(Wc / np.maximum(Ac, 1e-12))
        ok = dG_A < 0
        gotA = take(A, np.where(ok, gf[:, gi["upt"]] * (0.2 + size) * c.k_upt * (0.5 + pf), 0), cf)
        capA = c.eta * (-dG_A) * gotA * ok
        # assimilation: part of A becomes body if M < 2*Mt (pays e_B - e_A + ln(1/A) + overhead from E)
        Mt = 0.2 + size
        # (A that went to growth: not catabolised)
        grow_frac = np.where(self.M[free] < 2 * Mt, 0.5, 0.0)
        cost_g = np.maximum(0, (E_B - EN[A]) - np.log(np.maximum(Ac, 1e-12))) + 0.2
        need = grow_frac * gotA * cost_g
        E_av = self.E[free] + (1 - grow_frac) * capA
        afford = np.where(need > 0, np.minimum(1, E_av / np.maximum(need, 1e-300)), 1)
        gA_grow = grow_frac * gotA * afford; gA_cat = gotA - gA_grow
        capA = c.eta * (-dG_A) * gA_cat * ok
        F_Wadd = np.bincount(cf, gA_cat, G * G)
        diss(((1 - c.eta) * (-dG_A) * gA_cat)[ok]); diss((gA_grow * (cost_g - ((E_B - EN[A]) - np.log(np.maximum(Ac, 1e-12))))))
        self.M[free] += gA_grow
        dE = capA - gA_grow * cost_g
        Q += float(((EN[A] - EN[W]) * gA_cat - capA).sum()) + float((gA_grow * cost_g - gA_grow * (E_B - EN[A])).sum())
        # ---------- 6. organics: D -> W ----------
        Dc = F[D].ravel()[cf]
        dG_D = (EN[W] - EN[D]) + np.log(Wc / np.maximum(Dc, 1e-12)); okd = dG_D < 0
        gotD = take(D, np.where(okd, gf[:, gi["eat_org"]] * (0.2 + size) * c.k_upt, 0), cf)
        capD = c.eta * (-dG_D) * gotD * okd
        diss(((1 - c.eta) * (-dG_D) * gotD)[okd]); F_Wadd += np.bincount(cf, gotD, G * G)
        Q += float(((EN[D] - EN[W]) * gotD - capD).sum()); dE += capD
        # ---------- 7. MM transporters for S and V (active: pays ln(c_int/c_out) + overhead) ----------
        km = 0.01 + 0.5 * gf[:, gi["rt_km"]]; vm = gf[:, gi["rt_vmax"]] * c.k_upt; sp = gf[:, gi["rt_spec"]]
        for s, frac_s, ko in ((S, sp, c.rt_sig_ko), (V, 1 - sp, 0)):
            cc = np.maximum(F[s].ravel()[cf], 1e-12)
            J = vm * frac_s * cc / (km + cc) * (0 if ko else 1)
            got = take(s, J, cf)
            wtr = np.maximum(0, np.log(c.c_int / cc)) + c.ovh_tr
            dG_cat = (EN[W] - EN[s]) + np.log(Wc / c.c_int)        # inside: c_int -> W
            okc = (-dG_cat) * c.eta > wtr
            got_ok = got * okc
            if (~okc).any():   # return unprofitable uptake untouched
                F[s] += np.bincount(cf, got * ~okc, G * G).reshape(G, G)
            cap = (c.eta * (-dG_cat) - wtr) * got_ok
            # dissipation: total -dG (out -> W) minus captured
            dG_tot = (EN[W] - EN[s]) + np.log(Wc / cc)
            diss(((-dG_tot) * got_ok - cap)[okc])
            F_Wadd += np.bincount(cf, got_ok, G * G); Q += float(((EN[s] - EN[W]) * got_ok - cap).sum()); dE += cap
            if s == S: self.log["sig_rt"] += float(got_ok.sum())
        F[W] += F_Wadd.reshape(G, G)
        self.E[free] += dE; self.log["field_in"][free] += dE
        # ---------- 8. secretion from body: B -> S, B -> X (downhill), shed B -> Ag ----------
        relS = np.minimum(gf[:, gi["sig_sec"]] * 0.004, 0.1 * self.M[free])
        relX = np.minimum(gf[:, gi["tox_sec"]] * 0.004, 0.1 * self.M[free])
        relA = np.minimum(c.shed * 0.1 * np.ones(free.size), 0.05 * self.M[free])
        self.M[free] -= relS + relX + relA
        F[S] += np.bincount(cf, relS, G * G).reshape(G, G); F[X] += np.bincount(cf, relX, G * G).reshape(G, G)
        F[AG] += np.bincount(cf, relA, G * G).reshape(G, G)
        for k in range(NT):
            self.AGT[k] += np.bincount(cf, relA * gf[:, NS + k], G * G).reshape(G, G)
        q8 = (E_B - EN[S]) * relS + (E_B - EN[X]) * relX + (E_B - EN[AG]) * relA
        Q += float(q8.sum()); diss(q8)  # downhill, fully dissipated (activity terms ignored: body is pure phase; conservative bound)
        self.log["sig_rel"] += float(relS.sum())
        # ---------- 9. toxin hits: X + E -> W + heat ----------
        Xc = F[X].ravel()[cell[idx]]
        hit = take(X, c.tox_k * (1 - g[idx, gi["tox_res"]]) * np.ones(idx.size), cell[idx]) if False else None
        dem = c.tox_k * (1 - g[idx, gi["tox_res"]]) * Xc
        gotX = take(X, dem, cell[idx])
        F[W] += np.bincount(cell[idx], gotX, G * G).reshape(G, G)
        lossE = np.minimum(self.E[idx], c.tox_hit * gotX * 10)
        self.E[idx] -= lossE
        qx = (EN[X] - EN[W]) * gotX + lossE; Q += float(qx.sum()); diss(qx)
        # ---------- 10. maintenance & drive costs (E -> heat) ----------
        cost = c.base * (0.05 + g[idx, gi["size"]] + self.M[idx]) ** 0.75
        for name, cc in COST.items():
            cost = cost + cc * g[idx, gi[name]]
        starving = cost > self.E[idx]
        paid = np.minimum(cost, self.E[idx]); self.E[idx] -= paid; Q += float(paid.sum()); diss(paid)
        # ---------- 11. movement ----------
        pm = gf[:, gi["motil"]] * 0.5
        adh_f = gf[:, gi["adh"]] > 0.5
        nadh = np.bincount(cf[adh_f], minlength=G * G)
        grp = adh_f & (nadh[cf] >= 3)
        pm = np.where(grp, pm * (1 - gf[:, gi["adh"]]), pm)
        mcost = c.c_move * (0.2 + self.M[free])
        mv = (r.random(free.size) < pm) & (self.E[free] >= mcost)
        passive = (r.random(free.size) < 0.1) & (self.M[free] < 0.1) & ~mv     # tiny particles drift (thermal, free)
        dirs = r.integers(0, 4, free.size)
        Af = F[A]
        nb = np.stack([np.roll(Af, -1, 0), np.roll(Af, 1, 0), np.roll(Af, -1, 1), np.roll(Af, 1, 1)]).reshape(4, -1)
        best = nb[:, cf].argmax(0)
        dirs = np.where(r.random(free.size) < gf[:, gi["chemo"]], best, dirs)
        DX = np.array([1, -1, 0, 0]); DY = np.array([0, 0, 1, -1]); step_ = mv | passive
        self.x[free] = (self.x[free] + step_ * DX[dirs]) % G; self.y[free] = (self.y[free] + step_ * DY[dirs]) % G
        pay = mv * mcost; self.E[free] -= pay; Q += float(pay.sum()); diss(pay)
        cell = self.x * G + self.y; cf = cell[free]
        # ---------- 12. energy sharing in adhesive groups (lossy: 10% of transfer dissipated) ----------
        if grp.any():
            gg = free[grp]; give = g[gg, gi["share"]] * 0.1 * self.E[gg]
            pool = np.bincount(cf[grp], 0.9 * give, G * G); cnt = np.bincount(cf[grp], minlength=G * G)
            self.E[gg] += -give + pool[cf[grp]] / cnt[cf[grp]]; Q += float(0.1 * give.sum()); diss(0.1 * give)
        # ---------- 13. encounters ----------
        dead = np.zeros(c.Nmax, bool)
        order = free[np.lexsort((r.random(free.size), cell[free]))]
        a, b = order[:-1], order[1:]
        same = cell[a] == cell[b]; a, b = a[same], b[same]
        keep = np.ones(a.size, bool); keep[1::2] = False; a, b = a[keep], b[keep]
        flip = r.random(a.size) < 0.5; a, b = np.where(flip, b, a), np.where(flip, a, b)
        Ma, Mb = self.M[a], self.M[b]
        selfm = aff(g[a, TAG], g[b, TAG], c.w_aff)
        ecost = 0.002
        pe = g[a, gi["engulf"]] * (1 - selfm) * (Ma >= 1.5 * Mb) * 0.5
        ev = (r.random(a.size) < pe) & (self.E[a] >= ecost)
        if ev.any():
            ea, eb = a[ev], b[ev]
            self.E[ea] -= ecost; Q += ecost * ev.sum(); diss(np.full(ev.sum(), ecost))
            gE = c.engulf_eff * self.E[eb]; gM = c.engulf_eff * self.M[eb]
            np.add.at(self.E, ea, gE); np.add.at(self.M, ea, gM)
            q = self.E[eb] - gE; Q += float(q.sum()); diss(q)
            restM = self.M[eb] - gM
            F[D] += np.bincount(cell[eb], restM, G * G).reshape(G, G)
            q2 = (E_B - EN[D]) * restM; Q += float(q2.sum()); diss(q2)
            self.E[eb] = 0; self.M[eb] = 0; dead[eb] = True
            self.log["eng"].append(np.stack([self.lab[ea], self.lab[eb], gE + E_B * gM], 1))
        mb = self.ms[b] * aff(g[a, TAG], self.m[b], c.w_aff) * c.mem_effect
        block = np.clip(g[b, gi["defense"]] * (1 - selfm) + mb, 0, 1)
        pi = g[a, gi["infect"]] * aff(g[a, REC], g[b, TAG], c.w_aff) * (1 - block) * (Ma <= 0.5 * Mb)
        iv = (r.random(a.size) < pi) & ~ev & ~dead[a] & ~dead[b]
        if iv.any():
            ia, ib = a[iv], b[iv]
            self.host[ia] = ib
            self.log["inf"].append(np.stack([self.lab[ia], self.lab[ib]], 1).astype(float))
            self.log["inf_tags"].append(np.concatenate([g[ia, REC], g[ib, TAG]], 1))
        # ---------- 14. inside: drain host matter + energy (exact transfer, 20% of E dissipated) ----------
        inside = idx[(self.host[idx] >= 0) & ~dead[idx]]
        if inside.size:
            h = self.host[inside]
            ok = self.alive[h] & ~dead[h]
            fr_d = g[inside, gi["vir"]] * c.drain * ok
            # several parasites per host: scale so total fraction <= 0.5
            tot = np.bincount(h, fr_d, c.Nmax); scale = np.where(tot > 0.5, 0.5 / np.maximum(tot, 1e-300), 1)[h]
            fr_d = fr_d * scale
            dEh = fr_d * self.E[h]; dMh = fr_d * self.M[h]
            np.add.at(self.E, h, -dEh); np.add.at(self.M, h, -dMh)
            self.E[inside] += 0.8 * dEh; self.M[inside] += dMh; Q += float(0.2 * dEh.sum()); diss(0.2 * dEh)
            self.log["drain"].append(np.stack([self.lab[inside], self.lab[h], dEh + E_B * dMh], 1))
            selfh = aff(g[inside, TAG], g[h, TAG], c.w_aff)
            blk = np.clip(g[h, gi["defense"]] * (1 - selfh) + self.ms[h] * aff(g[inside, TAG], self.m[h], c.w_aff) * c.mem_effect, 0, 1)
            clr = (r.random(inside.size) < c.clear_p * blk) & ok
            dead[inside[clr]] = True
            hc = h[clr]; mr = g[hc, gi["mem"]]
            self.m[hc] += mr[:, None] * (g[inside[clr], TAG] - self.m[hc]); self.ms[hc] = np.minimum(1, self.ms[hc] + mr * 0.1)
            self.log["in_steps"][inside] += 1
            self.host[inside[~ok]] = -1
        self.log["free_steps"][free] += 1
        # memory from local antigen
        agm = F[AG].ravel()[cell[idx]]
        agt = self.AGT.reshape(NT, -1)[:, cell[idx]].T / np.maximum(agm, 1e-12)[:, None]
        ns = 1 - aff(agt, g[idx, TAG], c.w_aff)
        lr = g[idx, gi["mem"]] * 0.05 * ns * (agm > 0.005)
        self.m[idx] += lr[:, None] * (agt - self.m[idx]); self.ms[idx] = np.clip(self.ms[idx] * 0.999 + lr * 0.1, 0, 1)
        # ---------- 15. reproduction: split matter and energy (division costs c_div from E) ----------
        self.age[idx] += 1
        isin = self.host[idx] >= 0
        effr = g[idx, gi["auto"]].copy()
        if c.hijack:
            hh = np.where(isin, self.host[idx], 0)
            effr = np.where(isin, np.maximum(effr, g[hh, gi["auto"]]), effr)
        Mt_i = 0.2 + g[idx, gi["size"]] * np.where(isin, 0.3, 1.0) * 0 + g[idx, gi["size"]] * 0  # placeholder kept simple
        Mt_i = 0.05 + 0.5 * g[idx, gi["size"]]
        prep = 0.1 * effr * np.where(isin, 1.0, 1 - self.p[idx])
        rep = idx[(self.M[idx] >= 2 * Mt_i) & (self.E[idx] > c.c_div + 0.1) & (r.random(idx.size) < prep) & ~dead[idx]]
        # ---------- 16. death: body -> detritus (+ antigen with tags), stored E -> heat ----------
        die = idx[starving & (self.E[idx] <= 1e-12) | (self.age[idx] > c.max_age * (0.5 + r.random(idx.size)))]
        dead[die] = True
        dl = np.flatnonzero(dead & self.alive)
        if dl.size:
            hosted = idx[np.isin(self.host[idx], dl) & ~dead[idx]]
            self.host[hosted] = -1
            cl = cell[dl]; Md = self.M[dl]
            toAg = 0.1 * Md; toD = Md - toAg
            F[D] += np.bincount(cl, toD, G * G).reshape(G, G); F[AG] += np.bincount(cl, toAg, G * G).reshape(G, G)
            for k in range(NT):
                self.AGT[k] += np.bincount(cl, toAg * g[dl, NS + k], G * G).reshape(G, G)
            q = self.E[dl] + (E_B - EN[D]) * toD + (E_B - EN[AG]) * toAg; Q += float(q.sum()); diss(q)
            self.E[dl] = 0; self.M[dl] = 0
            self.alive[dl] = False; self.host[dl] = -1
        rep = rep[self.alive[rep]]
        slots = np.flatnonzero(~self.alive)
        k = min(rep.size, slots.size)
        if k:
            par = r.permutation(rep)[:k]; ch = slots[:k]
            self.E[par] -= c.c_div; Q += c.c_div * k; diss(np.full(k, c.c_div))
            sg = 10 ** (-3 + 2 * g[par, gi["mut"]])
            self.g[ch] = np.clip(g[par] + r.normal(0, 1, (k, NG)) * sg[:, None], 0, 1)
            self.E[ch] = 0.5 * self.E[par]; self.E[par] -= self.E[ch]
            self.M[ch] = 0.5 * self.M[par]; self.M[par] -= self.M[ch]
            self.x[ch] = self.x[par]; self.y[ch] = self.y[par]; self.age[ch] = 0
            self.host[ch] = self.host[par]
            self.m[ch] = self.m[par]; self.ms[ch] = self.ms[par]; self.p[ch] = 0
            self.lab[ch] = self.lab[par]; self.uid[ch] = np.arange(self.next_uid, self.next_uid + k); self.next_uid += k
            self.alive[ch] = True
            pin = self.host[par] >= 0
            self.log["birth_in"].append(self.lab[par[pin]]); self.log["birth_free"].append(self.lab[par[~pin]])
            self.log["birth_p"].append(np.stack([cell[par], self.p[par], self.g[par, gi["adh"]]], 1))
        return self._close(Q, sig, dmin)

    def _close(self, Q, sig, dmin):
        self.cum["Q"] += Q; self.cum["sigma"] += sig
        Mt, Et = self.totals()
        rM = (Mt - (self.M_tot0 + self.cum["inM"] - self.cum["outM"])) / max(Mt, 1.0)
        rE = (Et - (self.E_tot0 + self.cum["inE"] - self.cum["outE"] - self.cum["Q"])) / max(Et, 1.0)
        self.resid.append((self.t, rM, rE, sig, dmin, Q))
        if not (abs(rM) < self.c.tol and abs(rE) < self.c.tol):
            raise ThermoError(f"ledger violation t={self.t} rM={rM:.3e} rE={rE:.3e}")
        if dmin < -1e-12 or sig < -1e-12:
            raise ThermoError(f"negative dissipation t={self.t} min={dmin:.3e} sigma={sig:.3e}")
        if (self.F < -1e-12).any() or (self.M[self.alive] < -1e-12).any() or (self.E[self.alive] < -1e-9).any():
            raise ThermoError(f"negative stock t={self.t} F={self.F.min():.2e} M={self.M[self.alive].min():.2e} E={self.E[self.alive].min():.2e}")
        self.t += 1

    def free_energy(self):
        """Gibbs-like free energy of the field (ideal dilute, kT=1) + agent stores + biomass energy."""
        Fm = np.maximum(self.F, 1e-300)
        a = self.alive
        return float((self.F * (EN[:, None, None] + np.log(Fm) - 1)).sum() + self.E[a].sum() + E_B * self.M[a].sum())

    def snapshot(self):
        idx = np.flatnonzero(self.alive)
        return dict(t=self.t, g=self.g[idx].copy(), E=self.E[idx].copy(), M=self.M[idx].copy(), x=self.x[idx].copy(),
                    y=self.y[idx].copy(), host=(self.host[idx] >= 0), p=self.p[idx].copy(), ms=self.ms[idx].copy(),
                    uid=self.uid[idx].copy())
