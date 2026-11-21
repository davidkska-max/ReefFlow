"""Fit photon output (PPF) and beam (cos^m) of fixtures to Reef Builders' in-air PAR grids.

Model = the ReefFlow emitter model (reefflow.light.fixtures.Distribution): a uniformly emitting
rectangle of the housing size, radiant intensity per unit flux I(theta)/Phi of a cos^m
distribution. In air, a horizontal sensor at depth h below the emitting face gets
E = sum_i (PPF/N) * Inorm(theta_i) * cos(theta_i) / r_i^2   (sensor tilt = emission angle).
Data: peak (centre) at 23.25/24 and 30 in, mean over the 3x3 ft grid (169 pts, 3 in spacing)
and the central 2x2 ft (81 pts). Unknowns: PPF, m. Least squares in log(PAR)."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reefflow.light.fixtures import Distribution  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IN = 0.0254


def field(ppf, m, L, W, h, pts, n=12):
    d = Distribution("cosine_power", m=m)
    xs = (np.arange(n) + 0.5) / n * L - L / 2
    ys = (np.arange(n) + 0.5) / n * W - W / 2
    E = np.zeros(len(pts))
    for x in xs:
        for y in ys:
            dx, dy = pts[:, 0] - x, pts[:, 1] - y
            r2 = dx * dx + dy * dy + h * h
            c = h / np.sqrt(r2)
            E += d.normalised_intensity(np.arccos(c)) * c / r2
    return ppf / (n * n) * E           # umol/s / m^2 -> umol/m^2/s


def grid(half, step=3 * IN):
    v = np.arange(-half, half + 1e-9, step)
    X, Y = np.meshgrid(v, v)
    return np.c_[X.ravel(), Y.ravel()]


G3, G2, C = grid(18 * IN), grid(12 * IN), np.zeros((1, 2))
assert len(G3) == 169 and len(G2) == 81

FIX = {   # name: (L, W, watts measured, data [(h in, kind, PAR)])
    "XR30 G6 Pro": (0.30, 0.18, 215.9, [(24, "peak", 173.2), (30, "peak", 111.7), (24, "g3", 124.9), (30, "g3", 92.4),
                                         (24, "g2", 150.9), (30, "g2", 103.0), (23.25, "peak", 172.0)]),
    "CoralCare Gen2": (0.46, 0.41, 167.9, [(24, "peak", 201.5), (30, "peak", 134.0), (24, "g3", 109.9), (30, "g3", 91.6),
                                            (24, "g2", 148.02), (30, "g2", 113.33), (23.25, "peak", 201.4)]),
    "Blade Coral Grow 21": (0.536, 0.07, 42.2, [(24, "peak", 76.0), (30, "peak", 49.7), (24, "g3", 42.3), (30, "g3", 33.4),
                                                 (24, "g2", 56.7), (30, "g2", 41.2), (23.25, "peak", 73.19)]),
}


def predict(ppf, m, L, W, data):
    out = []
    for h, kind, _ in data:
        pts = {"peak": C, "g3": G3, "g2": G2}[kind]
        out.append(field(ppf, m, L, W, h * IN, pts).mean())
    return np.array(out)


for name, (L, W, watts, data) in FIX.items():
    meas = np.array([d[2] for d in data])
    res = least_squares(lambda p: np.log(predict(math.exp(p[0]), math.exp(p[1]), L, W, data)) - np.log(meas),
                        x0=[math.log(300), math.log(1.0)])
    ppf, m = math.exp(res.x[0]), math.exp(res.x[1])
    pred = predict(ppf, m, L, W, data)
    beam = Distribution("cosine_power", m=m).beam_angle
    print(f"{name}: PPF {ppf:.0f} umol/s, {ppf / watts:.2f} umol/J at {watts} W, m {m:.2f} (beam {math.degrees(beam):.0f} deg)")
    for (h, kind, v), p in zip(data, pred):
        print(f"   {kind:4s} at {h:5.2f} in: measured {v:6.1f}  model {p:6.1f}  ({100 * (p / v - 1):+.0f} %)")
    print(f"   rms log error {np.sqrt(np.mean(res.fun ** 2)) * 100:.1f} %")
