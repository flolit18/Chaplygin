"""Cas isotrope : solution exacte Omega = Omega_0, r = r_0 + a (Omega_y, -Omega_x) t."""
import numpy as np
import pytest

from chaplygin import ChaplyginSphere
from chaplygin.euler import simulation as euler_sim
from chaplygin.quaternion import simulation as quat_sim

M, A = 1.0, 0.1
SPHERE = ChaplyginSphere(m=M, a=A, Ib=[0.4 * M * A**2] * 3)
R0 = np.zeros(2)
ALPHA0 = np.array([0.8, 0.7, 0.0])
OMEGA0 = np.array([-0.31, 0.10, 5.30])


@pytest.mark.parametrize("sim", [euler_sim, quat_sim])
def test_isotropic_exact_solution(sim):
    res = sim.simulate(SPHERE, R0, ALPHA0, OMEGA0, 1e-3, 2000)
    r_exact = R0 + A * np.outer(res.t, [OMEGA0[1], -OMEGA0[0]])
    assert res.valid
    assert np.abs(res.Omega - OMEGA0).max() < 1e-9
    assert np.abs(res.r - r_exact).max() < 1e-9
    assert np.abs(res.friction).max() < 1e-9
