"""Particle animation data (D-849): the flow-field volume sent to the browser."""
import numpy as np

from reefflow.domain import FLUID, SOLID
from reefflow.ui import results as RS


def _run(tmp_path, shape=(40, 20, 16)):
    rng = np.random.default_rng(1)
    U = rng.normal(size=shape + (3,)).astype(np.float32)
    flags = np.full(shape, FLUID, np.int8)
    flags[10:14, 5:9, :6] = SOLID                       # a rock
    np.savez(tmp_path / "fields.npz", mean_u=U, mean_speed=np.linalg.norm(U, axis=-1), tke=np.zeros(shape, np.float32),
             tags=np.zeros(shape, np.uint8), flags=flags, origin=np.array([0.001, 0.002, 0.003]), dx=np.float64(0.005))
    return U, flags


def test_velocity_volume_full_and_strided(tmp_path):
    U, flags = _run(tmp_path)
    v = RS.velocity_volume(tmp_path)
    assert v["stride"] == 1 and v["shape"] == [40, 20, 16] and v["dx"] == 0.005
    assert np.isnan(v["U"][11, 6, 2]).all()                                  # rock -> NaN
    assert np.array_equal(v["U"][0, 0, 0], U[0, 0, 0])                         # water -> exact
    v2 = RS.velocity_volume(tmp_path, max_nodes=2000)                           # 12800 nodes -> stride 2
    assert v2["stride"] == 2 and v2["shape"] == [20, 10, 8] and v2["dx"] == 0.01
    assert np.array_equal(v2["U"][3, 4, 5], U[6, 8, 10])
    assert v2["origin"] == [0.001, 0.002, 0.003]
