"""Steady mean water age on the time-averaged field (PLAN C5).

Equation (age a in seconds, incompressible mean flow U, turbulent diffusivity D):

    div(U a) - div(D grad a) = 1,

with age-0 water entering at the return-nozzle source nodes, water leaving at
its local age through the overflow sink nodes, and zero flux through walls and
the free-slip lid. Finite volumes on the LBM node grid (one cell per fluid
node), first-order upwind advection and central diffusion.

Steps
1. Face fluxes F = U_face dx^2 from the averaged LBM velocity. The LBM mean
   field is only approximately divergence-free (weak compressibility, finite
   averaging), so F is projected to satisfy the discrete continuity equation
   with the pump volume sources exactly: L phi = div F - q, F <- F - grad phi
   (graph Laplacian, algebraic multigrid). The correction is reported.
2. D = nu + (nu_t,resolved + nu_t,SGS) / Sc_t, with Sc_t = 0.7:
     nu_t,resolved = max(0, -R'_ij S_ij / (2 S_kl S_kl))  (least-squares Boussinesq
                     fit of the resolved deviatoric Reynolds stress to the mean strain),
     nu_t,SGS      = (C_s dx)^2 |S|  of the mean strain (a lower bound).
   Both are smoothed over 3x3x3 cells.
3. Solve the sparse system (GMRES + incomplete-LU preconditioner).

Built-in check: summing the discrete equations over all cells gives
sum(outflow x age) = V exactly, i.e. the flow-weighted mean age at the overflow
equals V / Q (the nominal residence time) for ANY flow field. The solver
reports this ratio; it must be 1 to solver tolerance.

Normalised age a / (V/Q): 1 = what a perfectly mixed tank would have
everywhere; > 1 = water that is refreshed more slowly than average (stagnant);
< 1 = short-circuit paths from the return to the overflow.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

from .. import sources as S

SC_T = 0.7          # turbulent Schmidt number (ASSUMPTIONS A-302)
NU = 0.94e-6


def strain_rate(mean_u, dx):
    """Mean strain-rate tensor S_ij (nx, ny, nz, 6): xx, yy, zz, xy, xz, yz (1/s).
    Solid nodes hold zero velocity, so gradients next to walls see the no-slip value."""
    g = [np.gradient(mean_u[..., i], dx) for i in range(3)]     # g[i][j] = d u_i / d x_j
    return np.stack([g[0][0], g[1][1], g[2][2], 0.5 * (g[0][1] + g[1][0]),
                     0.5 * (g[0][2] + g[2][0]), 0.5 * (g[1][2] + g[2][1])], axis=-1)


def _ss(Sij):
    return (Sij[..., 0] ** 2 + Sij[..., 1] ** 2 + Sij[..., 2] ** 2
            + 2 * (Sij[..., 3] ** 2 + Sij[..., 4] ** 2 + Sij[..., 5] ** 2))


def eddy_viscosity(mean_u, reynolds, fluid, dx, cs=0.10):
    """(nu_t total, nu_t resolved, nu_t SGS) arrays, m^2/s, smoothed over 3x3x3 fluid cells."""
    Sij = strain_rate(mean_u, dx)
    ss = _ss(Sij)
    tr = (reynolds[..., 0] + reynolds[..., 1] + reynolds[..., 2]) / 3.0
    Rd = reynolds.copy()
    Rd[..., 0:3] -= tr[..., None]
    w = np.array([1, 1, 1, 2, 2, 2.0])
    num = -(Rd * Sij * w).sum(axis=-1)
    nu_res = np.where(ss > 1e-12, np.clip(num / (2 * np.maximum(ss, 1e-12)), 0, None), 0.0)
    nu_sgs = (cs * dx) ** 2 * np.sqrt(2 * ss)

    def smooth(a):
        m = fluid.astype(float)
        num_ = ndimage.uniform_filter(np.where(fluid, a, 0.0), 3, mode="constant")
        den = ndimage.uniform_filter(m, 3, mode="constant")
        return np.where(fluid, num_ / np.maximum(den, 1e-12), 0.0)

    nu_res, nu_sgs = smooth(nu_res), smooth(nu_sgs)
    return nu_res + nu_sgs, nu_res, nu_sgs


def _faces(fluid):
    """Arrays (c, nb, axis) of face-neighbour fluid pairs, c < nb along +axis."""
    out = []
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        both = fluid[tuple(sl0)] & fluid[tuple(sl1)]
        c = np.argwhere(both)
        nb = c.copy()
        nb[:, a] += 1
        out.append((c, nb, a))
    return out


def solve_age(model, mean_u, nu_t, *, sc_t=SC_T, tol=1e-9) -> dict:
    """Water age for a tank model: volume sources from its return/overflow source rows."""
    if model.spec is None or model.overflow_index is None:
        raise ValueError("water age needs a return pump and an overflow")
    tab, cells = model.spec.table, model.spec.cells
    m = tab[:, S.S_MASS] != 0
    out = solve_age_arrays(model.fluid_mask, mean_u, NU + nu_t / sc_t, model.dx,
                           cells[m], tab[m, S.S_MASS] * model.lu.C_Q, tol=tol)
    out["report"].update(sc_t=sc_t, nu_t_mean=float(nu_t[model.fluid_mask].mean()))
    return out


def solve_age_arrays(fluid, mean_u, D, dx, src_cells, src_q, *, tol=1e-9) -> dict:
    """Core solver. fluid: bool (nx,ny,nz); mean_u (nx,ny,nz,3) m/s; D (nx,ny,nz) m^2/s;
    src_cells (n,3) nodes with volume sources src_q (m^3/s; > 0 age-0 inflow, < 0 outflow)."""
    t0 = time.perf_counter()
    shape = fluid.shape
    idx = -np.ones(shape, dtype=np.int64)
    N = int(fluid.sum())
    idx[fluid] = np.arange(N)
    A_f = dx * dx
    q = np.zeros(N)
    src_cells = np.asarray(src_cells)
    np.add.at(q, idx[src_cells[:, 0], src_cells[:, 1], src_cells[:, 2]], np.asarray(src_q, float))
    Q = float(q[q > 0].sum())
    if Q <= 0 or abs(q.sum()) > 1e-9 * Q:
        raise ValueError("age needs balanced positive inflow and outflow volume sources")
    cs, nbs, F, G = [], [], [], []
    for c, nb, a in _faces(fluid):
        ic = idx[c[:, 0], c[:, 1], c[:, 2]]
        inb = idx[nb[:, 0], nb[:, 1], nb[:, 2]]
        uf = 0.5 * (mean_u[c[:, 0], c[:, 1], c[:, 2], a] + mean_u[nb[:, 0], nb[:, 1], nb[:, 2], a])
        df = 0.5 * (D[c[:, 0], c[:, 1], c[:, 2]] + D[nb[:, 0], nb[:, 1], nb[:, 2]])
        cs.append(ic); nbs.append(inb); F.append(uf * A_f); G.append(df * A_f / dx)
    c = np.concatenate(cs); nb = np.concatenate(nbs); F = np.concatenate(F); G = np.concatenate(G)

    # 1. projection to discrete continuity with the pump sources
    div = np.bincount(c, F, N) - np.bincount(nb, F, N)
    r = div - q
    one = np.ones_like(F)
    L = sp.coo_matrix((np.concatenate([one, one, -one, -one]),
                       (np.concatenate([c, nb, c, nb]), np.concatenate([c, nb, nb, c]))), shape=(N, N)).tocsr()
    import pyamg
    # ground node 0 (the Laplacian is singular up to a constant); r sums to zero
    L = L.tolil()
    L[0, :] = 0.0
    L[0, 0] = 1.0
    L = L.tocsr()
    rr = r.copy()
    rr[0] = 0.0
    ml = pyamg.smoothed_aggregation_solver(L, symmetry="nonsymmetric")
    phi = ml.solve(rr, tol=1e-12, maxiter=500, accel="gmres")
    F2 = F - (phi[c] - phi[nb])
    div2 = np.bincount(c, F2, N) - np.bincount(nb, F2, N)
    fs = max(np.abs(F).sum(), np.abs(F2).sum(), 1e-300)
    proj = {"rel_divergence_before": float(np.abs(r).sum() / fs),
            "rel_divergence_after": float(np.abs(div2 - q).sum() / fs),
            "rel_flux_change": float(np.abs(F2 - F).sum() / fs)}

    # 2. age system
    pos = F2 > 0
    Fp = np.where(pos, F2, 0.0)
    Fn = np.where(pos, 0.0, -F2)
    sink = np.clip(-q, 0, None)
    rows = np.concatenate([c, nb, nb, c, c, nb, c, nb])
    cols = np.concatenate([c, c, nb, nb, c, nb, nb, c])
    vals = np.concatenate([Fp, -Fp, Fn, -Fn, G, G, -G, -G])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr() + sp.diags(sink)
    b = np.full(N, dx ** 3)
    t1 = time.perf_counter()
    if N <= 50_000:
        a = spla.spsolve(A.tocsc(), b)            # small systems: direct
    else:
        # algebraic multigrid (nonsymmetric smoothed aggregation) as GMRES preconditioner
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="nonsymmetric", strength="evolution")
        a = b / A.diagonal()
        for _ in range(8):                      # restarted AMG-GMRES on the true residual
            rres = b - A @ a
            if np.linalg.norm(rres) <= tol * np.linalg.norm(b):
                break
            a = a + ml.solve(rres, tol=1e-6, maxiter=200, accel="gmres")
        if np.linalg.norm(b - A @ a) > 10 * tol * np.linalg.norm(b):
            ilu = spla.spilu(A.tocsc(), drop_tol=1e-5, fill_factor=10)    # robust fallback
            M = spla.LinearOperator(A.shape, ilu.solve)
            a, info = spla.gmres(A, b, x0=a, M=M, rtol=tol, atol=0.0, restart=60, maxiter=2000)
            if info != 0:
                raise RuntimeError(f"age solver did not converge (gmres info {info})")
    res = float(np.linalg.norm(A @ a - b) / np.linalg.norm(b))
    V = N * dx ** 3
    tau = V / Q
    out_age = float((sink * a).sum() / sink.sum())
    age = np.full(shape, np.nan)
    age[fluid] = a
    rep = {
        "volume_m3": V, "return_flow_m3s": Q, "tau_nominal_s": tau,
        "outlet_mean_age_s": out_age, "outlet_age_over_tau": out_age / tau,
        "mean_age_s": float(a.mean()), "mean_age_over_tau": float(a.mean() / tau),
        "p95_age_over_tau": float(np.percentile(a, 95) / tau), "max_age_over_tau": float(a.max() / tau),
        "min_age_s": float(a.min()), "residual": res, "projection": proj,
        "D_mean": float(D[fluid].mean()),
        "unknowns": N, "solve_time_s": time.perf_counter() - t0, "linear_solve_time_s": time.perf_counter() - t1,
    }
    return {"age": age, "report": rep}
