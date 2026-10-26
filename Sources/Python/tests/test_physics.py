"""Tests du modele.  A lancer depuis Python/ :  python -m pytest tests"""
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import physics

ALPHA = np.array([0.4, -0.7, 1.3])


def test_det_E_is_cos_theta():
    assert np.isclose(np.linalg.det(physics.E_body(ALPHA)), np.cos(ALPHA[1]))


def test_E_space_is_R_E():
    R = physics.rotation_matrix(ALPHA)
    assert np.allclose(physics.E_space(ALPHA), R @ physics.E_body(ALPHA))


def test_reduced_matrix_is_inertia_at_P():
    s = ChaplyginSphere(m=1.0, a=0.1, Ib=[3e-3, 4e-3, 5e-3])
    Es = physics.E_space(ALPHA)
    assert np.allclose(s.reduced_matrix(ALPHA), Es.T @ s.inertia_P(ALPHA) @ Es)
