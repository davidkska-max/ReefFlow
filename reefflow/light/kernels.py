"""Taichi kernel of the Monte Carlo forward photon tracer.

(No ``from __future__ import annotations`` here: Taichi kernels need real annotations.)

One GPU/CPU thread traces one photon from emission to absorption or escape.
All photons start with weight 1; the host converts tallies to physical units
by multiplying with (total PPF) / (photons traced).

Scene (tank coordinates, m): the water occupies the box [0, L] x [0, W] x
[z_sand, z_s]. The four side panes are glass slabs of thickness g just
outside that box, the sand bed is the plane z = z_sand, the water surface is
the plane z = z_s and the open top of the tank (rim) is at z = z_rim. Rock is
a signed-distance field (sdf < 0 inside).

Physics per event
    air -> water surface   Fresnel (unpolarised) + Snell with an optional
                           randomly tilted facet normal (RMS slope, Cox-Munk-like
                           isotropic Gaussian slopes); reflected photons are lost.
    in water               Beer-Lambert: weight x exp(-Kd s) over each straight
                           path segment (Kd per spectral bin).
    water -> air (below)   Fresnel / total internal reflection; transmitted photons escape.
    side walls             clear/black: Fresnel water->glass, explicit propagation
                           in the slab (n_g), Fresnel/TIR at glass->air (clear) or
                           absorption at the painted outer face (black); coralline:
                           Lambertian reflection at the inner face with an albedo.
    sand, rock             tallied as incident photon flux; then absorbed, or
                           re-emitted Lambertian with probability = albedo
                           (Russian roulette keeps all weights equal).

Tallies (all atomic adds of weight w and w^2, so the host can form per-cell
standard errors)
    stack     downward crossings of horizontal planes at the node heights of a
              node grid (meter-equivalent planar PAR at every node, PLAN C6);
              sum of w only (noise from 1/sqrt(sum w), D-410)
    rock      incident flux binned by the nearest node (surface-normal irradiance
              after division by the patch area)
    sand      incident flux on the sand plane per node column
    plan      downward crossings of user planes (plan maps), own xy grid
    probes    downward crossings of small horizontal squares (calibration)
    counters  energy balance (replicated rows to limit atomic contention)
"""
import math

import taichi as ti

# float parameters P[...]
(P_L, P_W, P_ZS, P_ZSAND, P_RIM, P_G, P_NW, P_NG, P_ALB_SAND, P_ALB_ROCK, P_ALB_COR, P_ROUGH,
 P_ZFLOOR, P_SDF_H, P_SDF_OX, P_SDF_OY, P_SDF_OZ, P_GOX, P_GOY, P_GOZ, P_GDX, P_PX0, P_PY0, P_PDX,
 P_RX0, P_RY0, P_RDX) = range(27)
NPF = 32
# integer parameters IP[...]
(I_NEMIT, I_NTAB, I_NBIN, I_HASROCK, I_SNX, I_SNY, I_SNZ, I_GNX, I_GNY, I_GNZ, I_NPLAN, I_PNX, I_PNY,
 I_NPROBE, I_WALL0, I_WALL1, I_WALL2, I_WALL3, I_MAXEV, I_STACK, I_RNX, I_RNY, I_HASWATER, I_BOUNCE,
 I_ROUGHMAP) = range(25)
NPI = 32
# counters (photon weight)
(C_EMIT, C_MISS, C_SREFL, C_ENTER, C_WABS, C_SAND, C_ROCK, C_GLASS, C_BLACK, C_CORAL, C_ESC, C_TERM,
 C_DX, C_DY, C_DZ, C_FLOOR, C_OTHER, C_UP, C_SANDIN, C_ROCKIN, C_WALLIN, C_GLASSIN) = range(22)
NC = 24
COUNTER_NAMES = ["emitted", "missed_tank", "surface_reflected", "entered_water", "absorbed_water",
                 "absorbed_sand", "absorbed_rock", "lost_through_glass", "absorbed_black_wall",
                 "absorbed_coralline", "escaped_surface", "terminated_event_cap", "entry_dir_x",
                 "entry_dir_y", "entry_dir_z", "reached_floor_air", "lost_other", "emitted_upward",
                 "incident_sand", "incident_rock", "incident_walls", "entered_glass"]
REPLICAS = 4096

F1 = ti.types.ndarray(dtype=ti.f32, ndim=1)
F2 = ti.types.ndarray(dtype=ti.f32, ndim=2)
F3 = ti.types.ndarray(dtype=ti.f32, ndim=3)
I1 = ti.types.ndarray(dtype=ti.i32, ndim=1)
# tally arrays: dtype chosen at run time (fp64 where the backend has fp64 atomics, D-409)
T1 = ti.types.ndarray(ndim=1)
T2 = ti.types.ndarray(ndim=2)
T3 = ti.types.ndarray(ndim=3)
TWO_PI = 2.0 * math.pi
BIG = 1e30


# ----------------------------------------------------------------------
# random numbers: per-photon PCG (deterministic, identical on every backend)
# ----------------------------------------------------------------------
@ti.func
def lcg(s, inc):
    """PCG32 state step; every photon has its own odd increment (its own stream)."""
    return s * ti.u32(747796405) + inc


@ti.func
def hash32(x):
    """PCG-RXS-M-XS integer hash (bijective, full avalanche)."""
    st = x * ti.u32(747796405) + ti.u32(2891336453)
    w = (ti.bit_shr(st, ti.bit_shr(st, 28) + ti.u32(4)) ^ st) * ti.u32(277803737)
    return ti.bit_shr(w, 22) ^ w


@ti.func
def u01(s):
    w = (ti.bit_shr(s, ti.bit_shr(s, 28) + ti.u32(4)) ^ s) * ti.u32(277803737)
    w = ti.bit_shr(w, 22) ^ w
    return ti.cast(ti.bit_shr(w, 8), ti.f32) * (1.0 / 16777216.0)


@ti.func
def gauss(s0, inc):
    """Two LCG steps -> one standard normal (Box-Muller). Returns (value, state)."""
    s = lcg(s0, inc)
    a = u01(s)
    s = lcg(s, inc)
    b = u01(s)
    return ti.sqrt(-2.0 * ti.log(1.0 - a)) * ti.cos(TWO_PI * b), s


# ----------------------------------------------------------------------
# optics
# ----------------------------------------------------------------------
@ti.func
def fresnel(cos_i, eta):
    """Unpolarised Fresnel reflectance; eta = n1/n2; 1 under total internal reflection."""
    sin2t = eta * eta * (1.0 - cos_i * cos_i)
    R = 1.0
    if sin2t < 1.0:
        cos_t = ti.sqrt(1.0 - sin2t)
        rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
        rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
        R = 0.5 * (rs * rs + rp * rp)
    return R


@ti.func
def refract(d, n, eta):
    """Snell refraction of unit d at a surface with unit normal n facing the incoming side."""
    cos_i = -d.dot(n)
    sin2t = eta * eta * (1.0 - cos_i * cos_i)
    return (eta * d + (eta * cos_i - ti.sqrt(ti.max(1.0 - sin2t, 0.0))) * n).normalized()


@ti.func
def onb(n):
    """Orthonormal basis (b1, b2) perpendicular to unit n (Duff et al. 2017)."""
    sgn = 1.0
    if n.z < 0.0:
        sgn = -1.0
    a = -1.0 / (sgn + n.z)
    b = n.x * n.y * a
    b1 = ti.Vector([1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x])
    b2 = ti.Vector([b, sgn + n.y * n.y * a, -n.y])
    return b1, b2


@ti.func
def lambert_dir(n, s0, inc):
    """Cosine-weighted direction about unit n. Returns (d, state)."""
    s = lcg(s0, inc)
    u1 = u01(s)
    s = lcg(s, inc)
    u2 = u01(s)
    r = ti.sqrt(u1)
    ph = TWO_PI * u2
    b1, b2 = onb(n)
    d = (r * ti.cos(ph) * b1 + r * ti.sin(ph) * b2 + ti.sqrt(ti.max(1.0 - u1, 0.0)) * n).normalized()
    return d, s


@ti.func
def facet_normal(d, sig, s0, inc):
    """Upward unit facet normal of a rough water surface (isotropic Gaussian slopes, total
    RMS slope sig), sampled with probability proportional to the facet's projected area
    seen by the ray, |d.m| / (m_z |d_z|) (rejection sampling). Flat if sig == 0."""
    m = ti.Vector([0.0, 0.0, 1.0])
    s = s0
    if sig > 0.0:
        tan_d = ti.sqrt(ti.max(1.0 - d.z * d.z, 0.0)) / ti.max(ti.abs(d.z), 1e-3)
        rmax = 1.0 + 3.0 * sig * tan_d
        done = 0
        tries = 0
        while done == 0 and tries < 64:
            tries += 1
            gx, s = gauss(s, inc)
            gy, s = gauss(s, inc)
            c = ti.Vector([-gx * sig * 0.70710678, -gy * sig * 0.70710678, 1.0]).normalized()
            dm = d.dot(c)
            if dm * d.z > 0.0:                 # facet faces the incoming ray
                r = ti.abs(dm) / (c.z * ti.abs(d.z))
                s = lcg(s, inc)
                if u01(s) * rmax < r:
                    m = c
                    done = 1
    return m, s


# ----------------------------------------------------------------------
# rock SDF
# ----------------------------------------------------------------------
@ti.func
def sdf_eval(sdf: ti.template(), P: ti.template(), IP: ti.template(), p):
    h = P[P_SDF_H]
    q = (p - ti.Vector([P[P_SDF_OX], P[P_SDF_OY], P[P_SDF_OZ]])) / h
    nx = IP[I_SNX]
    ny = IP[I_SNY]
    nz = IP[I_SNZ]
    hi = ti.Vector([nx - 1.0, ny - 1.0, nz - 1.0])
    dv = ti.max(ti.max(-q, q - hi), 0.0)
    dout = dv.norm()
    val = 0.0
    if dout > 0.0:
        val = (dout + 1.0) * h
    else:
        i = ti.min(ti.max(ti.cast(ti.floor(q.x), ti.i32), 0), nx - 2)
        j = ti.min(ti.max(ti.cast(ti.floor(q.y), ti.i32), 0), ny - 2)
        k = ti.min(ti.max(ti.cast(ti.floor(q.z), ti.i32), 0), nz - 2)
        fx = q.x - i
        fy = q.y - j
        fz = q.z - k
        c00 = sdf[i, j, k] * (1 - fx) + sdf[i + 1, j, k] * fx
        c10 = sdf[i, j + 1, k] * (1 - fx) + sdf[i + 1, j + 1, k] * fx
        c01 = sdf[i, j, k + 1] * (1 - fx) + sdf[i + 1, j, k + 1] * fx
        c11 = sdf[i, j + 1, k + 1] * (1 - fx) + sdf[i + 1, j + 1, k + 1] * fx
        val = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz
    return val


@ti.func
def sdf_normal(sdf: ti.template(), P: ti.template(), IP: ti.template(), p):
    e = 0.5 * P[P_SDF_H]
    g = ti.Vector([sdf_eval(sdf, P, IP, p + ti.Vector([e, 0.0, 0.0])) - sdf_eval(sdf, P, IP, p - ti.Vector([e, 0.0, 0.0])),
                   sdf_eval(sdf, P, IP, p + ti.Vector([0.0, e, 0.0])) - sdf_eval(sdf, P, IP, p - ti.Vector([0.0, e, 0.0])),
                   sdf_eval(sdf, P, IP, p + ti.Vector([0.0, 0.0, e])) - sdf_eval(sdf, P, IP, p - ti.Vector([0.0, 0.0, e]))])
    n = ti.Vector([0.0, 0.0, 1.0])
    if g.norm() > 0.0:
        n = g.normalized()
    return n


@ti.func
def march(sdf: ti.template(), P: ti.template(), IP: ti.template(), p, d, tmax):
    """Distance along d to the rock surface (sphere tracing on the trilinear SDF, then
    bisection once a step lands inside); BIG if no hit before tmax."""
    h = P[P_SDF_H]
    eps_hit = 0.01 * h
    hmin = 0.05 * h
    t = 0.0
    hit = BIG
    phi = sdf_eval(sdf, P, IP, p)
    it = 0
    while it < 2048:
        it += 1
        if phi < eps_hit:
            hit = t
            break
        if t >= tmax:
            break
        tn = ti.min(t + ti.max(phi, hmin), tmax)
        phin = sdf_eval(sdf, P, IP, p + d * tn)
        if phin < 0.0:
            a = t
            b = tn
            for _ in range(24):
                mid = 0.5 * (a + b)
                if sdf_eval(sdf, P, IP, p + d * mid) < 0.0:
                    b = mid
                else:
                    a = mid
            hit = a
            break
        t = tn
        phi = phin
    return hit


# ----------------------------------------------------------------------
# tallies
# ----------------------------------------------------------------------
@ti.func
def tally_segment(p, d, tl, w, kdv, in_water, P: ti.template(), IP: ti.template(),
                  sw: ti.template(), pz: ti.template(), pw: ti.template(),
                  pw2: ti.template(), probes: ti.template(), qw: ti.template(), qw2: ti.template()):
    """Downward plane crossings of the straight segment p + d t, 0 <= t <= tl."""
    if d.z < 0.0:
        z1 = p.z
        z2 = p.z + d.z * tl
        # node-height stack (water only)
        if IP[I_STACK] == 1 and in_water == 1:
            oz = P[P_GOZ]
            dz = P[P_GDX]
            gnz = IP[I_GNZ]
            khi = ti.min(ti.cast(ti.floor((z1 - oz) / dz), ti.i32), gnz - 1)
            klo = ti.max(ti.cast(ti.floor((z2 - oz) / dz), ti.i32) + 1, 0)
            for k in range(klo, khi + 1):
                tc = (oz + k * dz - z1) / d.z
                i = ti.cast(ti.floor((p.x + d.x * tc - P[P_GOX]) / dz + 0.5), ti.i32)
                j = ti.cast(ti.floor((p.y + d.y * tc - P[P_GOY]) / dz + 0.5), ti.i32)
                if 0 <= i < IP[I_GNX] and 0 <= j < IP[I_GNY]:
                    # no sum of w^2 here (halves the atomics of the costliest tally): stack weights
                    # are never split, only attenuated (0.9-1), so rel. noise = 1/sqrt(sum w) (D-410)
                    sw[i, j, k] += w * ti.exp(-kdv * tc)
        # plan-map planes
        for l in range(IP[I_NPLAN]):
            zp = pz[l]
            if z2 < zp <= z1:
                tc = (zp - z1) / d.z
                i = ti.cast(ti.floor((p.x + d.x * tc - P[P_PX0]) / P[P_PDX]), ti.i32)
                j = ti.cast(ti.floor((p.y + d.y * tc - P[P_PY0]) / P[P_PDX]), ti.i32)
                if 0 <= i < IP[I_PNX] and 0 <= j < IP[I_PNY]:
                    wc = w * ti.exp(-kdv * tc)
                    pw[l, i, j] += wc
                    pw2[l, i, j] += wc * wc
        # probes
        for l in range(IP[I_NPROBE]):
            zp = probes[l, 2]
            if z2 < zp <= z1:
                tc = (zp - z1) / d.z
                if ti.abs(p.x + d.x * tc - probes[l, 0]) <= probes[l, 3] and \
                        ti.abs(p.y + d.y * tc - probes[l, 1]) <= probes[l, 3]:
                    wc = w * ti.exp(-kdv * tc)
                    qw[l] += wc
                    qw2[l] += wc * wc


# ----------------------------------------------------------------------
# main kernel
# ----------------------------------------------------------------------
@ti.kernel
def trace(n: ti.i32, seed: ti.u32, P: F1, IP: I1, emit: F2, ecdf: F1, prof: F2, scdf: F2, kd: F1,
          rough: F2, sdf: F3, sw: T3, rw: T3, rw2: T3, aw: T2, aw2: T2, pz: F1, pw: T3, pw2: T3,
          probes: F2, qw: T1, qw2: T1, C: T2):
    for ip in range(n):
        rep = ip % C.shape[0]
        inc = ti.cast(ip, ti.u32) * ti.u32(2) + ti.u32(1)
        s = hash32(hash32(ti.cast(ip, ti.u32)) ^ seed)
        # ---- emitter (probability proportional to its photon output) ----
        s = lcg(s, inc)
        u = u01(s)
        lo = 0
        hi = IP[I_NEMIT] - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if ecdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        ax = ti.Vector([emit[e, 3], emit[e, 4], emit[e, 5]])
        eu = ti.Vector([emit[e, 6], emit[e, 7], emit[e, 8]])
        ev = ti.Vector([emit[e, 9], emit[e, 10], emit[e, 11]])
        s = lcg(s, inc)
        a1 = u01(s)
        s = lcg(s, inc)
        a2 = u01(s)
        p = ti.Vector([emit[e, 0], emit[e, 1], emit[e, 2]]) + (2.0 * a1 - 1.0) * emit[e, 12] * eu \
            + (2.0 * a2 - 1.0) * emit[e, 13] * ev
        # ---- direction from the tabulated inverse CDF of mu = cos(theta) ----
        pr = ti.cast(emit[e, 14], ti.i32)
        s = lcg(s, inc)
        x = u01(s) * (IP[I_NTAB] - 1)
        it = ti.min(ti.cast(x, ti.i32), IP[I_NTAB] - 2)
        fr = x - it
        mu = prof[pr, it] * (1.0 - fr) + prof[pr, it + 1] * fr
        s = lcg(s, inc)
        ph = TWO_PI * u01(s)
        st = ti.sqrt(ti.max(1.0 - mu * mu, 0.0))
        d = (st * ti.cos(ph) * eu + st * ti.sin(ph) * ev + mu * ax).normalized()
        # ---- wavelength bin ----
        kdv = kd[0]
        if IP[I_NBIN] > 1:
            s = lcg(s, inc)
            ub = u01(s)
            b = 0
            while b < IP[I_NBIN] - 1 and scdf[e, b] < ub:
                b += 1
            kdv = kd[b]
        w = 1.0
        C[rep, C_EMIT] += w
        medium = 0                       # 0 air, 1 water, 2 glass
        gface = 0
        alive = 1
        nev = 0
        L = P[P_L]
        W = P[P_W]
        zs = P[P_ZS]
        zsand = P[P_ZSAND]
        nw = P[P_NW]
        ng = P[P_NG]
        eps = 1e-4 * P[P_SDF_H] + 1e-7
        while alive == 1:
            nev += 1
            if nev > IP[I_MAXEV]:
                C[rep, C_TERM] += w
                alive = 0
            elif medium == 0:
                # ================= air =================
                if d.z >= -1e-9:
                    C[rep, C_UP] += w
                    alive = 0
                elif IP[I_HASWATER] == 0:
                    t = (P[P_ZFLOOR] - p.z) / d.z
                    tally_segment(p, d, t, w, 0.0, 0, P, IP, sw, pz, pw, pw2, probes, qw, qw2)
                    C[rep, C_FLOOR] += w
                    alive = 0
                else:
                    inside = 1
                    if p.z > P[P_RIM]:
                        tr = (P[P_RIM] - p.z) / d.z
                        xr = p.x + d.x * tr
                        yr = p.y + d.y * tr
                        if xr < 0.0 or xr > L or yr < 0.0 or yr > W:
                            inside = 0
                    t = (zs - p.z) / d.z
                    q = p + d * t
                    if q.x < 0.0 or q.x > L or q.y < 0.0 or q.y > W:
                        inside = 0
                    if inside == 0:
                        C[rep, C_MISS] += w
                        alive = 0
                    else:
                        sig = P[P_ROUGH]
                        if IP[I_ROUGHMAP] == 1:
                            ri = ti.min(ti.max(ti.cast((q.x - P[P_RX0]) / P[P_RDX], ti.i32), 0), IP[I_RNX] - 1)
                            rj = ti.min(ti.max(ti.cast((q.y - P[P_RY0]) / P[P_RDX], ti.i32), 0), IP[I_RNY] - 1)
                            sig = rough[ri, rj]
                        m, s = facet_normal(d, sig, s, inc)
                        cos_i = -d.dot(m)
                        R = fresnel(cos_i, 1.0 / nw)
                        s = lcg(s, inc)
                        if u01(s) < R:
                            C[rep, C_SREFL] += w
                            alive = 0
                        else:
                            d = refract(d, m, 1.0 / nw)
                            if d.z > 0.0:
                                d.z = -d.z
                            p = q
                            p.z = zs
                            medium = 1
                            C[rep, C_ENTER] += w
                            C[rep, C_DX] += w * d.x
                            C[rep, C_DY] += w * d.y
                            C[rep, C_DZ] += w * d.z
            elif medium == 1:
                # ================= water =================
                tb = BIG
                face = -1
                if d.x > 1e-12:
                    tt = (L - p.x) / d.x
                    if tt < tb:
                        tb = tt
                        face = 3
                elif d.x < -1e-12:
                    tt = -p.x / d.x
                    if tt < tb:
                        tb = tt
                        face = 2
                if d.y > 1e-12:
                    tt = (W - p.y) / d.y
                    if tt < tb:
                        tb = tt
                        face = 1
                elif d.y < -1e-12:
                    tt = -p.y / d.y
                    if tt < tb:
                        tb = tt
                        face = 0
                if d.z > 1e-12:
                    tt = (zs - p.z) / d.z
                    if tt < tb:
                        tb = tt
                        face = 5
                elif d.z < -1e-12:
                    tt = (zsand - p.z) / d.z
                    if tt < tb:
                        tb = tt
                        face = 4
                tb = ti.max(tb, 0.0)
                tev = tb
                evn = face
                if IP[I_HASROCK] == 1:
                    tr = march(sdf, P, IP, p, d, tb)
                    if tr < tb:
                        tev = tr
                        evn = 6
                tally_segment(p, d, tev, w, kdv, 1, P, IP, sw, pz, pw, pw2, probes, qw, qw2)
                wn = w * ti.exp(-kdv * tev)
                C[rep, C_WABS] += w - wn
                w = wn
                p = p + d * tev
                gnx = IP[I_GNX]
                gny = IP[I_GNY]
                gnz = IP[I_GNZ]
                gdx = P[P_GDX]
                if evn == 6:
                    # ---- rock ----
                    nrm = sdf_normal(sdf, P, IP, p)
                    i = ti.cast(ti.floor((p.x - P[P_GOX]) / gdx + 0.5), ti.i32)
                    j = ti.cast(ti.floor((p.y - P[P_GOY]) / gdx + 0.5), ti.i32)
                    k = ti.cast(ti.floor((p.z - P[P_GOZ]) / gdx + 0.5), ti.i32)
                    if 0 <= i < gnx and 0 <= j < gny and 0 <= k < gnz:
                        rw[i, j, k] += w
                        rw2[i, j, k] += w * w
                    C[rep, C_ROCKIN] += w
                    s = lcg(s, inc)
                    if IP[I_BOUNCE] == 1 and u01(s) < P[P_ALB_ROCK]:
                        d, s = lambert_dir(nrm, s, inc)
                        p = p + nrm * (0.05 * P[P_SDF_H])
                    else:
                        C[rep, C_ROCK] += w
                        alive = 0
                elif evn == 4:
                    # ---- sand ----
                    p.z = zsand
                    i = ti.cast(ti.floor((p.x - P[P_GOX]) / gdx + 0.5), ti.i32)
                    j = ti.cast(ti.floor((p.y - P[P_GOY]) / gdx + 0.5), ti.i32)
                    if 0 <= i < gnx and 0 <= j < gny:
                        aw[i, j] += w
                        aw2[i, j] += w * w
                    C[rep, C_SANDIN] += w
                    s = lcg(s, inc)
                    if IP[I_BOUNCE] == 1 and u01(s) < P[P_ALB_SAND]:
                        d, s = lambert_dir(ti.Vector([0.0, 0.0, 1.0]), s, inc)
                        p.z = zsand + eps
                    else:
                        C[rep, C_SAND] += w
                        alive = 0
                elif evn == 5:
                    # ---- water surface from below ----
                    p.z = zs
                    sig = P[P_ROUGH]
                    if IP[I_ROUGHMAP] == 1:
                        ri = ti.min(ti.max(ti.cast((p.x - P[P_RX0]) / P[P_RDX], ti.i32), 0), IP[I_RNX] - 1)
                        rj = ti.min(ti.max(ti.cast((p.y - P[P_RY0]) / P[P_RDX], ti.i32), 0), IP[I_RNY] - 1)
                        sig = rough[ri, rj]
                    m, s = facet_normal(d, sig, s, inc)
                    cos_i = d.dot(m)
                    R = fresnel(cos_i, nw)
                    s = lcg(s, inc)
                    if u01(s) < R:
                        d = d - 2.0 * d.dot(m) * m
                        if d.z > 0.0:
                            d.z = -d.z
                        p.z = zs - eps
                    else:
                        C[rep, C_ESC] += w
                        alive = 0
                elif evn >= 0:
                    # ---- side wall ----
                    cond = IP[I_WALL0 + evn]
                    a = 1
                    if evn >= 2:
                        a = 0
                    sgn = 1.0
                    if evn == 1 or evn == 3:
                        sgn = -1.0
                    n_in = ti.Vector([0.0, 0.0, 0.0])
                    n_in[a] = sgn
                    C[rep, C_WALLIN] += w
                    if cond == 2:
                        s = lcg(s, inc)
                        if IP[I_BOUNCE] == 1 and u01(s) < P[P_ALB_COR]:
                            d, s = lambert_dir(n_in, s, inc)
                            p = p + n_in * eps
                        else:
                            C[rep, C_CORAL] += w
                            alive = 0
                    else:
                        cos_i = -d.dot(n_in)
                        R = fresnel(cos_i, nw / ng)
                        s = lcg(s, inc)
                        if u01(s) < R:
                            d = d + 2.0 * cos_i * n_in
                        else:
                            d = refract(d, n_in, nw / ng)
                            medium = 2
                            gface = evn
                            C[rep, C_GLASSIN] += w
                else:
                    C[rep, C_OTHER] += w
                    alive = 0
            else:
                # ================= glass slab =================
                a = 1
                if gface >= 2:
                    a = 0
                sgn = 1.0
                cin = 0.0
                if gface == 1:
                    sgn = -1.0
                    cin = W
                elif gface == 3:
                    sgn = -1.0
                    cin = L
                cout = cin - sgn * P[P_G]
                da = d[a]
                if ti.abs(da) < 1e-7:
                    C[rep, C_OTHER] += w
                    alive = 0
                elif da * sgn < 0.0:
                    # toward the outer face
                    t = (cout - p[a]) / da
                    p = p + d * t
                    if IP[I_WALL0 + gface] == 1:
                        C[rep, C_BLACK] += w
                        alive = 0
                    else:
                        R = fresnel(ti.abs(da), ng)
                        s = lcg(s, inc)
                        if u01(s) < R:
                            d[a] = -d[a]
                        else:
                            C[rep, C_GLASS] += w
                            alive = 0
                else:
                    # toward the inner face (back into the water)
                    t = (cin - p[a]) / da
                    p = p + d * t
                    p[a] = cin
                    b = 1 - a
                    ext = W
                    if b == 0:
                        ext = L
                    if p.z < zsand or p.z > zs or p[b] < 0.0 or p[b] > ext:
                        C[rep, C_OTHER] += w
                        alive = 0
                    else:
                        R = fresnel(ti.abs(da), ng / nw)
                        s = lcg(s, inc)
                        if u01(s) < R:
                            d[a] = -d[a]
                        else:
                            n_g = ti.Vector([0.0, 0.0, 0.0])
                            n_g[a] = -sgn
                            d = refract(d, n_g, ng / nw)
                            medium = 1
