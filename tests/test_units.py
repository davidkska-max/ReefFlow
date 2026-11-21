import math

import pytest

from reefflow import units as U


def test_us_si_exact_factors():
    assert U.to_si(1.0, "in") == pytest.approx(0.0254, rel=1e-15)
    assert U.to_si(1.0, "gal") == pytest.approx(3.785411784e-3, rel=1e-15)
    # 1 GPH = 3.785411784 L / 3600 s
    assert U.to_si(1.0, "GPH") == pytest.approx(1.0515032733e-6, rel=1e-9)
    assert U.to_si(1000.0, "L/h") == pytest.approx(1000e-3 / 3600)
    assert U.to_si(1.0, "lb") == pytest.approx(0.45359237, rel=1e-15)


@pytest.mark.parametrize("unit", ["in", "ft", "mm", "GPH", "GPM", "L/h", "gal", "cm/s", "lb", "ft_head"])
def test_round_trip(unit):
    assert U.from_si(U.to_si(123.456, unit), unit) == pytest.approx(123.456, rel=1e-14)


def test_unknown_unit():
    with pytest.raises(ValueError):
        U.to_si(1.0, "furlong")


def test_75_gallon_tank_volume():
    # 48 x 18 x 21 in external, 1/2 in glass: internal 47 x 17 x 20.5 in
    v = U.to_si(47, "in") * U.to_si(17, "in") * U.to_si(20.5, "in")
    # independent check: 1 US gallon is defined as exactly 231 cubic inches
    assert U.from_si(v, "gal") == pytest.approx(47 * 17 * 20.5 / 231, rel=1e-12)


def test_conversion_factors_dimensional_consistency():
    lu = U.LatticeUnits(dx=0.005, dt=2.5e-4, rho0=1025.0)
    assert lu.C_U == pytest.approx(20.0)
    assert lu.C_nu == pytest.approx(0.005**2 / 2.5e-4)
    assert lu.C_F == pytest.approx(lu.rho0 * lu.C_A)                 # N/m^3 = kg/m^3 * m/s^2
    assert lu.C_P == pytest.approx(lu.rho0 * lu.C_U**2)              # Pa = kg/m^3 (m/s)^2
    assert lu.C_Q == pytest.approx(lu.C_U * lu.dx**2)                # m^3/s = m/s * m^2
    assert lu.C_J == pytest.approx(lu.rho0 * lu.C_Q**2 / lu.dx**2)   # rho Q^2 / A


def test_reynolds_number_invariant():
    lu = U.LatticeUnits.from_velocity(dx=0.005, u_phys=1.0, u_lat=0.1)
    u, L, nu = 0.3, 0.5, 9.4e-7
    re_phys = u * L / nu
    re_lat = lu.velocity_to_lattice(u) * lu.length_to_cells(L) / lu.nu_to_lattice(nu)
    assert re_lat == pytest.approx(re_phys, rel=1e-12)


def test_acoustic_scaling():
    lu = U.LatticeUnits.from_velocity(dx=0.005, u_phys=1.0, u_lat=0.1)
    assert lu.dt == pytest.approx(5e-4)
    assert lu.velocity_to_lattice(1.0) == pytest.approx(0.1)
    assert lu.sound_speed == pytest.approx(math.sqrt(1 / 3) * 10.0)


def test_diffusive_scaling_tau():
    lu = U.LatticeUnits.from_tau(dx=1e-3, nu_phys=1e-6, tau=0.8)
    assert lu.tau_from_nu(1e-6) == pytest.approx(0.8)


def test_force_round_trip_and_momentum_flux():
    lu = U.LatticeUnits.from_velocity(dx=0.005, u_phys=1.0)
    f = 123.0  # N/m^3
    assert lu.force_density_to_physical(lu.force_density_to_lattice(f)) == pytest.approx(f)
    # a jet of Q through area A has momentum flux rho Q^2/A; lattice value consistent with lattice Q, A
    Q, A = 1e-3, 2e-3
    J = lu.rho0 * Q**2 / A
    Q_l, A_l = lu.flow_to_lattice(Q), A / lu.dx**2
    assert lu.momentum_flux_to_lattice(J) == pytest.approx(1.0 * Q_l**2 / A_l, rel=1e-12)


def test_pressure_from_density():
    lu = U.LatticeUnits.from_velocity(dx=0.01, u_phys=1.0)
    # p = cs^2 * (rho - 1) * rho0 * (dx/dt)^2
    assert lu.pressure_from_density(1.003) == pytest.approx(0.003 / 3 * lu.rho0 * 10.0**2)


def test_check_refuses_unsafe_parameters():
    lu = U.LatticeUnits.from_velocity(dx=0.005, u_phys=1.0, u_lat=0.1)
    rep = lu.check(u_max_phys=1.0, nu_phys=9.4e-7)
    assert rep["u_max_lat"] == pytest.approx(0.1)
    assert rep["tau_molecular"] > 0.5
    with pytest.raises(U.LatticeParameterError):
        lu.check(u_max_phys=2.0, nu_phys=9.4e-7)          # u_lat = 0.2 > 0.1
    rep = lu.check(u_max_phys=2.0, nu_phys=9.4e-7, strict=False)
    assert rep["problems"]


def test_invalid_construction():
    with pytest.raises(U.LatticeParameterError):
        U.LatticeUnits(dx=0.0, dt=1.0)
    with pytest.raises(U.LatticeParameterError):
        U.LatticeUnits.from_tau(dx=1e-3, nu_phys=1e-6, tau=0.5)
