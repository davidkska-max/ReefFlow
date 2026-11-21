"""Installation self-check: backend detection, a tiny solver run, and the
expected speed on this machine.

    python -m reefflow.selfcheck
"""
import sys
import time


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import reefflow  # noqa: F401  (disables Taichi's online version check)
    from reefflow import backend
    arch = backend.init()
    print(f"compute backend: {arch}" + ("" if arch in ("cuda", "vulkan", "metal") else
                                        "  (no GPU backend found: runs will be much slower)"))
    import numpy as np
    from reefflow.domain import Domain, SOLID
    from reefflow.solver import LBMSolver
    n = 64
    d = Domain((n, n, n))
    for f in ("x-", "x+", "y-", "y+", "z-", "z+"):
        d.set_face(f, SOLID)
    s = LBMSolver(d, lattice="D3Q19", collision="rr", nu=0.01, smagorinsky=0.1, body_force=(1e-6, 0, 0))
    m0 = s.total_mass()
    s.step(20)
    t = time.perf_counter()
    s.step(200)
    s.update_macros()
    _, u = s.macros()
    dt = time.perf_counter() - t
    ok = np.all(np.isfinite(u)) and abs(s.total_mass() / m0 - 1) < 1e-5
    mlups = 200 * n ** 3 / dt / 1e6
    print(f"solver test: {'OK' if ok else 'FAILED'}  ({mlups:.0f} million cell updates/s on a small grid; "
          "large grids run faster on a GPU)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
