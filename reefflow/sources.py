"""Volumetric source terms carried by SOURCE nodes (pumps, return nozzle, overflow).

Every SOURCE node owns one row of a *source table*; rows of the same pump share a
*pump state* row that the host updates as the run proceeds (pulse/wave modes,
inflow turbulence). Both tables are plain float arrays in lattice units.

Source table columns (per node)
    S_PUMP        pump index (row of the pump-state table)
    S_MASS        mass added per step, m0 (> 0 source, < 0 sink; lattice units)
    S_U0..+2      velocity of the added mass, u0 (lattice units)
    S_F0..+2      body force on the node, f0 (lattice units)
    S_PHI0..+6    inflow-turbulence mode shapes phi_k at this node (k = 0..6)

Pump-state columns (per pump)
    P_SF          force scale s_F(t)  (= s(t)^2: momentum flux scales with Q^2)
    P_SM          mass/velocity scale s_m(t)  (= s(t))
    P_I           turbulence intensity I
    P_A0..+6      unit-variance random mode amplitudes a_k(t)
    P_LAT..+2     lateral random vector l(t) (perpendicular to the pump axis, RMS 1)

Applied per node at each step (delta = sum_k a_k phi_k):
    force          F  = s_F [ f0 max(0, 1 + I delta) + I |f0| l ]
    mass m0 > 0    adds  m = s_m m0  of fluid in equilibrium at velocity
                   u_s = s_m [ u0 (1 + I delta) + I |u0| l ],
                   i.e. mass rate rho Q and momentum rate rho Q u0 = rho Q^2 / A.
    mass m0 < 0    removes the fraction |m|/rho of every population, i.e. removes
                   mass |m| together with its own momentum (no spurious force).
"""
from dataclasses import dataclass

import numpy as np

S_PUMP, S_MASS, S_U0, S_F0, S_PHI0 = 0, 1, 2, 5, 8
N_MODES = 7
S_NCOL = S_PHI0 + N_MODES          # 15

P_SF, P_SM, P_I, P_A0, P_LAT = 0, 1, 2, 3, 3 + N_MODES
P_NCOL = P_LAT + 3                  # 13


@dataclass
class SourceSpec:
    cells: np.ndarray       # (N, 3) int node indices
    table: np.ndarray       # (N, S_NCOL) float
    n_pumps: int

    def __post_init__(self):
        self.cells = np.asarray(self.cells, dtype=np.int64).reshape(-1, 3)
        self.table = np.asarray(self.table, dtype=np.float64).reshape(-1, S_NCOL)
        if len(self.cells) != len(self.table):
            raise ValueError("cells and table must have the same number of rows")
        if len(self.cells) == 0:
            raise ValueError("empty source specification")
        if len(np.unique(self.cells, axis=0)) != len(self.cells):
            raise ValueError("a node appears in more than one source row")
        p = self.table[:, S_PUMP]
        if np.any(p < 0) or np.any(p >= self.n_pumps) or np.any(p != np.round(p)):
            raise ValueError("pump index out of range")

    def default_state(self) -> np.ndarray:
        st = np.zeros((self.n_pumps, P_NCOL))
        st[:, P_SF] = 1.0
        st[:, P_SM] = 1.0
        return st

    # totals at the default (unit) pump state, in lattice units
    def mass_rate(self) -> float:
        return float(self.table[:, S_MASS].sum())

    def force_total(self) -> np.ndarray:
        return self.table[:, S_F0:S_F0 + 3].sum(axis=0)

    def momentum_rate(self) -> np.ndarray:
        """Momentum added per step by mass sources (sum m0 u0 over m0 > 0)."""
        m = np.clip(self.table[:, S_MASS], 0.0, None)
        return (m[:, None] * self.table[:, S_U0:S_U0 + 3]).sum(axis=0)
