"""Tests des representations de l'orientation."""
import numpy as np

from chaplygin import rotations

RNG = np.random.default_rng(0)
ALPHAS = RNG.uniform([-np.pi, -1.5, -np.pi], [np.pi, 1.5, np.pi], size=(100, 3))


def test_quat_from_euler_gives_same_rotation():
    for alpha in ALPHAS:
        q = rotations.quat_from_euler(alpha)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(rotations.quat_to_matrix(q), rotations.rotation_matrix(alpha))


def test_euler_from_quat_roundtrip():
    for alpha in ALPHAS:
        q = rotations.quat_from_euler(alpha)
        assert np.allclose(rotations.euler_from_quat(q), alpha)


def test_quat_mul_is_composition():
    q = rotations.quat_from_euler(ALPHAS[0])
    p = rotations.quat_from_euler(ALPHAS[1])
    R_qp = rotations.quat_to_matrix(rotations.quat_mul(q, p))
    assert np.allclose(R_qp, rotations.quat_to_matrix(q) @ rotations.quat_to_matrix(p))
