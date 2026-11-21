"""Taichi LBM kernels, generated per configuration and cached.

Kernels take Taichi *ndarrays* (runtime-shaped), so one compiled kernel set
serves every grid size with the same configuration (lattice, collision, LES on/off,
feature switches, periodicity). See solver.py for the algorithm description.

Storage convention: shifted populations g_q = f_q - w_q.
"""
import taichi as ti

from . import lattice as lat_mod
from .domain import FLUID, SOLID, FREESLIP, INLET, OUTLET, SOURCE
from . import sources as S

CS2 = 1.0 / 3.0
SQRT2 = 1.4142135623730951

I3 = ti.types.ndarray(dtype=ti.i8, ndim=3)
I32 = ti.types.ndarray(dtype=ti.i32, ndim=3)


def float_type():
    """Floating-point type of the current Taichi runtime (set by backend.init(precision=...))."""
    return ti.f64 if ti.lang.impl.current_cfg().default_fp == ti.f64 else ti.f32


def array_types(fp=None):
    """(F4, F3, V3, F2) ndarray annotations for float type fp."""
    fp = fp or float_type()
    return (ti.types.ndarray(dtype=fp, ndim=4), ti.types.ndarray(dtype=fp, ndim=3),
            ti.types.ndarray(dtype=ti.types.vector(3, fp), ndim=3), ti.types.ndarray(dtype=fp, ndim=2))


# fp32 annotations (helpers in stats.py / verification that always run in fp32 builds)
F4, F3, V3, F2 = array_types(ti.f32)

_cache: dict = {}


class KernelSet:
    pass


def get(lattice: str, collision: str, smag: bool, force_field: bool, moving_walls: bool,
        freeslip: bool, periodic: tuple, sponge: bool = False, sources: bool = False) -> KernelSet:
    key = (lattice.upper(), collision, bool(smag), bool(force_field), bool(moving_walls),
           bool(freeslip), tuple(bool(p) for p in periodic), float_type(), bool(sponge), bool(sources))
    if key not in _cache:
        _cache[key] = _build(*key)
    return _cache[key]


def _build(lattice, collision, smag, force_field, moving_walls, freeslip, periodic, fp, sponge, sources):
    F4, F3, V3, F2 = array_types(fp)
    L = lat_mod.get(lattice)
    Q = L.Q
    C = [tuple(int(v) for v in row) for row in L.c]
    W = [float(v) for v in L.w]
    OPP = [int(v) for v in L.opp]
    MIR = [[int(v) for v in row] for row in L.mirror]
    K1 = L.K1.tolist()
    K2 = L.K2.tolist()
    rr = collision == "rr"
    n3 = len(L.basis3) if rr else 0
    N3V = max(n3, 1)
    K3 = L.K3.tolist() if rr else [[] for _ in range(Q)]
    B3 = [[(float(c),) + tuple(lat_mod.TRIPLES[n]) for c, n in combo] for combo in L.basis3]
    PAIRS = list(lat_mod.PAIRS)
    PX, PY, PZ = periodic
    EPS = 1e-15

    # ------------------------------------------------------------------
    @ti.func
    def wrap(v, n, per: ti.template()):
        r = v
        if ti.static(per):
            if r < 0:
                r += n
            elif r >= n:
                r -= n
        else:
            if r < 0 or r >= n:
                r = -1
        return r

    @ti.func
    def moments(g):
        drho = 0.0
        j = ti.Vector([0.0, 0.0, 0.0])
        P = ti.Vector([0.0] * 6)
        for q in ti.static(range(Q)):
            drho += g[q]
            for a in ti.static(range(3)):
                if ti.static(C[q][a] != 0):
                    j[a] += C[q][a] * g[q]
            for p in ti.static(range(6)):
                a, b = ti.static(PAIRS[p])
                if ti.static(C[q][a] * C[q][b] != 0):
                    P[p] += C[q][a] * C[q][b] * g[q]
        return drho, j, P

    @ti.func
    def a2neq(drho, rho, u, P, F):
        a2 = ti.Vector([0.0] * 6)
        for p in ti.static(range(6)):
            a, b = ti.static(PAIRS[p])
            if ti.static(a == b):
                a2[p] = P[p] - CS2 * drho - rho * u[a] * u[a] + u[a] * F[a]
            else:
                a2[p] = P[p] - rho * u[a] * u[b] + 0.5 * (u[a] * F[b] + u[b] * F[a])
        return a2

    @ti.func
    def a3eq(rho, u):
        A = ti.Vector([0.0] * N3V)
        for k in ti.static(range(n3)):
            s = 0.0
            for t in ti.static(range(len(B3[k]))):
                coef, a, b, c = ti.static(B3[k][t])
                s += coef * u[a] * u[b] * u[c]
            A[k] = rho * s
        return A

    @ti.func
    def a3neq(u, a2):
        M = ti.Matrix([[a2[0], a2[3], a2[4]], [a2[3], a2[1], a2[5]], [a2[4], a2[5], a2[2]]])
        A = ti.Vector([0.0] * N3V)
        for k in ti.static(range(n3)):
            s = 0.0
            for t in ti.static(range(len(B3[k]))):
                coef, a, b, c = ti.static(B3[k][t])
                s += coef * (u[a] * M[b, c] + u[b] * M[a, c] + u[c] * M[a, b])
            A[k] = s
        return A

    @ti.func
    def E2of(rho, u):
        E = ti.Vector([0.0] * 6)
        for p in ti.static(range(6)):
            a, b = ti.static(PAIRS[p])
            E[p] = rho * u[a] * u[b]
        return E

    @ti.func
    def geq(q: ti.template(), drho, rho, u, E2, A3):
        r = W[q] * drho
        for a in ti.static(range(3)):
            if ti.static(K1[q][a] != 0.0):
                r += K1[q][a] * rho * u[a]
        for p in ti.static(range(6)):
            if ti.static(abs(K2[q][p]) > EPS):
                r += K2[q][p] * E2[p]
        for k in ti.static(range(n3)):
            if ti.static(abs(K3[q][k]) > EPS):
                r += K3[q][k] * A3[k]
        return r

    @ti.func
    def proj(q: ti.template(), a2, A3):
        r = 0.0
        for p in ti.static(range(6)):
            if ti.static(abs(K2[q][p]) > EPS):
                r += K2[q][p] * a2[p]
        for k in ti.static(range(n3)):
            if ti.static(abs(K3[q][k]) > EPS):
                r += K3[q][k] * A3[k]
        return r

    @ti.func
    def freeslip_pull(src: ti.template(), flags: ti.template(), aux: ti.template(), q: ti.template(), i, j, k, si, sj, sk, nx, ny, nz):
        val = src[OPP[q], i, j, k]       # fallback: bounce-back (edges/corners)
        a = ti.cast(aux[si, sj, sk], ti.i32)
        n0 = a // 9 - 1
        n1 = (a // 3) % 3 - 1
        n2 = a % 3 - 1
        cx, cy, cz = ti.static(C[q])
        if abs(n0) + abs(n1) + abs(n2) == 1:
            ti_ = i
            tj = j
            tk = k
            qs = 0
            if n0 != 0:
                tj = wrap(j - cy, ny, PY)
                tk = wrap(k - cz, nz, PZ)
                qs = MIR[0][q]
            elif n1 != 0:
                ti_ = wrap(i - cx, nx, PX)
                tk = wrap(k - cz, nz, PZ)
                qs = MIR[1][q]
            else:
                ti_ = wrap(i - cx, nx, PX)
                tj = wrap(j - cy, ny, PY)
                qs = MIR[2][q]
            if ti_ >= 0 and tj >= 0 and tk >= 0:
                tf = flags[ti_, tj, tk]
                if tf == FLUID or tf == INLET or tf == OUTLET or tf == SOURCE:
                    val = src[qs, ti_, tj, tk]
        return val

    @ti.func
    def source_terms(row, sidx: ti.template(), stab: ti.template(), pst: ti.template()):
        """(force, added mass, velocity of added mass) of source row (see sources.py)."""
        p = ti.cast(stab[row, S.S_PUMP], ti.i32)
        sF = pst[p, S.P_SF]
        sm = pst[p, S.P_SM]
        I = pst[p, S.P_I]
        delta = 0.0
        for m in ti.static(range(S.N_MODES)):
            delta += pst[p, S.P_A0 + m] * stab[row, S.S_PHI0 + m]
        lat = ti.Vector([pst[p, S.P_LAT], pst[p, S.P_LAT + 1], pst[p, S.P_LAT + 2]])
        f0 = ti.Vector([stab[row, S.S_F0], stab[row, S.S_F0 + 1], stab[row, S.S_F0 + 2]])
        u0 = ti.Vector([stab[row, S.S_U0], stab[row, S.S_U0 + 1], stab[row, S.S_U0 + 2]])
        Fs = sF * (f0 * ti.max(0.0, 1.0 + I * delta) + I * f0.norm() * lat)
        us = sm * (u0 * (1.0 + I * delta) + I * u0.norm() * lat)
        return Fs, sm * stab[row, S.S_MASS], us

    @ti.func
    def gather(src: ti.template(), flags: ti.template(), aux: ti.template(), u: ti.template(), i, j, k, nx, ny, nz):
        g = ti.Vector([0.0] * Q)
        for q in ti.static(range(Q)):
            cx, cy, cz = ti.static(C[q])
            si = i
            sj = j
            sk = k
            if ti.static(cx != 0):
                si = wrap(i - cx, nx, PX)
            if ti.static(cy != 0):
                sj = wrap(j - cy, ny, PY)
            if ti.static(cz != 0):
                sk = wrap(k - cz, nz, PZ)
            val = 0.0
            if si < 0 or sj < 0 or sk < 0:
                val = src[OPP[q], i, j, k]
            else:
                sf = flags[si, sj, sk]
                if sf == SOLID:
                    val = src[OPP[q], i, j, k]
                    if ti.static(moving_walls):
                        uw = u[si, sj, sk]
                        # Ladd moving wall: + 2 w rho_w (c.u_w)/cs^2 with rho_w = 1
                        val += 6.0 * W[q] * (cx * uw[0] + cy * uw[1] + cz * uw[2])
                else:
                    val = src[q, si, sj, sk]
                    if ti.static(freeslip):
                        if sf == FREESLIP:
                            val = freeslip_pull(src, flags, aux, q, i, j, k, si, sj, sk, nx, ny, nz)
            g[q] = val
        return g

    # ------------------------------------------------------------------
    @ti.kernel
    def init_eq(f: F4, flags: I3, rho_a: F3, u_a: V3):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        for i, j, k in ti.ndrange(nx, ny, nz):
            rho = rho_a[i, j, k]
            u = u_a[i, j, k]
            if flags[i, j, k] == SOLID:
                rho = 1.0
                u = ti.Vector([0.0, 0.0, 0.0])
            drho = rho - 1.0
            E2 = E2of(rho, u)
            A3 = a3eq(rho, u)
            for q in ti.static(range(Q)):
                f[q, i, j, k] = geq(q, drho, rho, u, E2, A3)

    @ti.kernel
    def step(src: F4, dst: F4, flags: I3, aux: I3, rho_a: F3, u_a: V3, force: V3, sp: F3,
             sidx: I32, stab: F2, pst: F2,
             tau0_in: fp, cs2: fp, gx: fp, gy: fp, gz: fp, store: ti.i32):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        G = ti.Vector([gx, gy, gz])
        for i, j, k in ti.ndrange(nx, ny, nz):
            fl = flags[i, j, k]
            active = fl == FLUID
            if ti.static(sources):
                active = active | (fl == SOURCE)
            if active:
                g = gather(src, flags, aux, u_a, i, j, k, nx, ny, nz)
                drho, jm, P = moments(g)
                F = G
                if ti.static(force_field):
                    F += force[i, j, k]
                msrc = 0.0
                usrc = ti.Vector([0.0, 0.0, 0.0])
                if ti.static(sources):
                    if fl == SOURCE:
                        Fs, mm, us = source_terms(sidx[i, j, k], sidx, stab, pst)
                        F += Fs
                        msrc = mm
                        usrc = us
                rho = 1.0 + drho
                u = (jm + 0.5 * F) / rho
                a2 = a2neq(drho, rho, u, P, F)
                tau0 = tau0_in
                tau_override = 0.0
                if ti.static(sponge):
                    spv = sp[i, j, k]
                    if spv >= 0.0:
                        tau0 += spv           # sponge layer: extra relaxation time (= 3 * extra nu)
                    else:
                        tau_override = -spv   # wall model (D-702): total relaxation time set directly
                omega = 1.0 / tau0
                if ti.static(smag):
                    qn = ti.sqrt(a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2]
                                 + 2.0 * (a2[3] * a2[3] + a2[4] * a2[4] + a2[5] * a2[5]))
                    tau = 0.5 * (tau0 + ti.sqrt(tau0 * tau0 + 18.0 * SQRT2 * cs2 * qn / rho))
                    omega = 1.0 / tau
                if ti.static(sponge):
                    if tau_override > 0.0:
                        omega = 1.0 / tau_override
                E2 = E2of(rho, u)
                A3 = a3eq(rho, u)
                T = ti.Vector([0.0] * 6)          # uF + Fu
                for p in ti.static(range(6)):
                    a, b = ti.static(PAIRS[p])
                    T[p] = u[a] * F[b] + u[b] * F[a]
                A3n = a3neq(u, a2)
                # mass source (m > 0): add m of fluid in equilibrium at usrc;
                # mass sink (m < 0): remove the fraction |m|/rho of every population
                mpos = ti.max(msrc, 0.0)
                frac = ti.min(msrc, 0.0) / rho
                E2s = E2of(mpos, usrc)
                A3s = a3eq(mpos, usrc)
                for q in ti.static(range(Q)):
                    ge = geq(q, drho, rho, u, E2, A3)
                    Fq = 0.0
                    for a in ti.static(range(3)):
                        if ti.static(K1[q][a] != 0.0):
                            Fq += K1[q][a] * F[a]
                    for p in ti.static(range(6)):
                        if ti.static(abs(K2[q][p]) > EPS):
                            Fq += K2[q][p] * T[p]
                    post = 0.0
                    if ti.static(collision == "bgk"):
                        post = ge + (1.0 - omega) * (g[q] - ge + 0.5 * Fq) + 0.5 * Fq
                    else:
                        post = ge + (1.0 - omega) * proj(q, a2, A3n) + 0.5 * Fq
                    if ti.static(sources):
                        post += frac * (post + W[q]) + geq(q, mpos, mpos, usrc, E2s, A3s)
                    dst[q, i, j, k] = post
                if store:
                    rho_a[i, j, k] = rho
                    u_a[i, j, k] = u

    @ti.kernel
    def open_bc(dst: F4, flags: I3, aux: I3, rho_a: F3, u_a: V3):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        for i, j, k in ti.ndrange(nx, ny, nz):
            fl = flags[i, j, k]
            if fl == INLET or fl == OUTLET:
                a = ti.cast(aux[i, j, k], ti.i32)
                ni = wrap(i + a // 9 - 1, nx, PX)
                nj = wrap(j + (a // 3) % 3 - 1, ny, PY)
                nk = wrap(k + a % 3 - 1, nz, PZ)
                gn = ti.Vector([0.0] * Q)
                for q in ti.static(range(Q)):
                    gn[q] = dst[q, ni, nj, nk]
                drho_n, jn, Pn = moments(gn)
                rho_n = 1.0 + drho_n
                u_n = jn / rho_n
                a2 = a2neq(drho_n, rho_n, u_n, Pn, ti.Vector([0.0, 0.0, 0.0]))
                rho_b = rho_n
                u_b = u_n
                if fl == INLET:
                    u_b = u_a[i, j, k]
                else:
                    rho_b = rho_a[i, j, k]
                drho_b = rho_b - 1.0
                E2 = E2of(rho_b, u_b)
                A3 = a3eq(rho_b, u_b)
                A3n = a3neq(u_b, a2)
                for q in ti.static(range(Q)):
                    dst[q, i, j, k] = geq(q, drho_b, rho_b, u_b, E2, A3) + proj(q, a2, A3n)

    @ti.kernel
    def macros(f: F4, flags: I3, rho_a: F3, u_a: V3, force: V3, sidx: I32, stab: F2, pst: F2,
               gx: fp, gy: fp, gz: fp):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        G = ti.Vector([gx, gy, gz])
        for i, j, k in ti.ndrange(nx, ny, nz):
            fl = flags[i, j, k]
            if fl == FLUID or fl == SOURCE:
                g = ti.Vector([0.0] * Q)
                for q in ti.static(range(Q)):
                    g[q] = f[q, i, j, k]
                drho, jm, P = moments(g)
                F = G
                if ti.static(force_field):
                    F += force[i, j, k]
                if ti.static(sources):
                    if fl == SOURCE:
                        Fs, mm, us = source_terms(sidx[i, j, k], sidx, stab, pst)
                        F += Fs
                rho = 1.0 + drho
                rho_a[i, j, k] = rho
                # f holds POST-collision populations, whose momentum is rho*u + F/2
                # (the collision added the full impulse F). Hence the minus sign.
                u_a[i, j, k] = (jm - 0.5 * F) / rho

    @ti.kernel
    def mass_lines(f: F4, flags: I3, out: F2):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        for i, j in ti.ndrange(nx, ny):
            s = 0.0
            for k in range(nz):
                fl = flags[i, j, k]
                if fl == FLUID or fl == SOURCE:
                    for q in ti.static(range(Q)):
                        s += f[q, i, j, k]
            out[i, j] = s

    @ti.kernel
    def set_bc_velocity(flags: I3, u_a: V3, src: V3):
        nx, ny, nz = flags.shape[0], flags.shape[1], flags.shape[2]
        for i, j, k in ti.ndrange(nx, ny, nz):
            fl = flags[i, j, k]
            if fl == INLET or fl == SOLID:
                u_a[i, j, k] = src[i, j, k]

    ks = KernelSet()
    ks.init_eq, ks.step, ks.open_bc, ks.macros = init_eq, step, open_bc, macros
    ks.mass_lines, ks.set_bc_velocity = mass_lines, set_bc_velocity
    return ks
