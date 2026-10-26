"""Les deux simulations donnent la meme trajectoire loin de la singularite."""
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin.controls import kinetic_energy, relative_drift
from chaplygin.euler import simulation as euler_sim
from chaplygin.quaternion import simulation as quat_sim

SPHERE = ChaplyginSphere(m=1.0, a=0.1, Ib=[1.5e-3, 8.5e-3, 9.5e-3])
R0 = np.zeros(2)


def test_same_trajectory_far_from_gimbal_lock():
    alpha0, Omega0 = np.array([0.8, 0.7, 0.0]), np.array([-0.31, 0.10, 5.30])
    res_e = euler_sim.simulate(SPHERE, R0, alpha0, Omega0, 1e-3, 2000)
    res_q = quat_sim.simulate(SPHERE, R0, alpha0, Omega0, 1e-3, 2000)
    assert res_e.valid and res_q.valid
    assert np.abs(res_e.r - res_q.r).max() < 1e-8
    assert np.abs(res_e.R - res_q.R).max() < 1e-8
    assert np.abs(res_e.friction - res_q.friction).max() < 1e-8


def test_quaternions_keep_energy_near_gimbal_lock():
    s = ChaplyginSphere(m=1.0, a=0.1, Ib=[3e-3, 4e-3, 5e-3])
    res = quat_sim.simulate(s, R0, [0.2, 0.3, 0.1], [5.0, 1.0, 2.0], 1e-3, 1000)
    assert res.valid
    assert relative_drift(kinetic_energy(s, res)).max() < 1e-8
    assert np.allclose(np.linalg.norm(res.q, axis=1), 1.0)
