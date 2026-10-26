"""
    Dynamique de la sphere de Chaplygin en quaternions (partie III.2).

    Etat Y = (q, omega) dans R^7 : q quaternion unitaire (orientation),
    omega vitesse angulaire dans le repere corps. Pas de singularite.
"""
import numpy as np

from ..rotations import quat_mul, quat_to_matrix
# e_z
E_Z = np.array([0.0, 0.0, 1.0])


def gamma(q):
    """
        Verticale vue depuis le corps : gamma = R(q)^T e_z
        On fait entrer vect(e_z)
    """
    return quat_to_matrix(q).T @ E_Z


def K(sphere, gam):
    """
        K(gamma) = I_b + m a^2 (Id - gamma gamma^T), symetrique definie positive
    """
    return sphere.Ib + sphere.m * sphere.a**2 * (np.eye(3) - np.outer(gam, gam))


def omega_dot(sphere, q, omega):
    """
        Resout K(gamma) omega_dot = -omega x (I_b omega)
    """
    b = -np.cross(omega, sphere.Ib @ omega)
    return np.linalg.solve(K(sphere, gamma(q)), b)


def q_dot(q, omega):
    """
        Cinematique quaternionique : q_dot = 1/2 q * (0, omega)
    """
    return 0.5 * quat_mul(q, np.concatenate([[0.0], omega]))


def f(sphere, Y):
    """
        Second membre Y_dot = (q_dot, omega_dot), avec Y = (q, omega)
    """
    q, omega = Y[:4], Y[4:]
    return np.concatenate([q_dot(q, omega), omega_dot(sphere, q, omega)])


# --- position et frottements

def r_dot(sphere, q, omega):
    """
        r_dot = a (Omega_y, -Omega_x), avec Omega = R(q) omega
    """
    Omega = quat_to_matrix(q) @ omega
    return sphere.a * np.array([Omega[1], -Omega[0]])


def friction(sphere, q, omega_dot):
    """
        (N_x, N_y) = m a (Omega_dot_y, -Omega_dot_x), avec Omega_dot = R(q) omega_dot
    """
    Omega_dot = quat_to_matrix(q) @ omega_dot
    return sphere.m * sphere.a * np.array([Omega_dot[1], -Omega_dot[0]])
