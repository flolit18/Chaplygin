"""
    Controle de la simulation : invariants et hypotheses de validite
"""
import numpy as np

from .physics import E_body, E_space


def kinetic_energy(sphere, alpha, alpha_dot):
    """
        T = 1/2 m |r_dot|^2 + 1/2 omega^T I_b omega (conservee)
    """
    r_dot = sphere.r_dot(alpha, alpha_dot)
    omega = E_body(alpha) @ alpha_dot
    return 0.5 * sphere.m * r_dot @ r_dot + 0.5 * omega @ sphere.Ib @ omega


def angular_momentum_P(sphere, alpha, alpha_dot):
    """
        L_P = I_P Omega, moment cinetique au point de contact dans le repere sol (conserve comme vecteur)
    """
    Omega = E_space(alpha) @ alpha_dot
    return sphere.inertia_P(alpha) @ Omega


def no_slip_ok(sphere, lam):
    """
        Non-glissement : |lambda| <= mu_s*m*g
    """
    return np.linalg.norm(lam) <= sphere.mu_s * sphere.m * sphere.g


def euler_ok(alphas, eps=1e-3):
    """
        Validite des angles d'Euler : |cos(theta)| > eps pour tous les etats
        donnes (etats intermediaires compris) et pas de changement de signe
    """
    c = np.cos(np.asarray(alphas)[:, 1])
    return bool(np.all(np.abs(c) > eps) and np.all(np.sign(c) == np.sign(c[0])))


def relative_drift(values):
    """
        Derive relative |q_k - q_0| / |q_0| d'une suite de quantites
        (scalaires ou vecteurs), pour chaque k. Le max donne la derive globale
    """
    q = np.asarray(values, dtype=float)
    q = q.reshape(len(q), -1)            # scalaires -> vecteurs de taille 1
    return np.linalg.norm(q - q[0], axis=1) / np.linalg.norm(q[0])
