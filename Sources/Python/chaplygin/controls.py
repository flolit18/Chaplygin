"""
    Controle de la simulation : invariants et hypotheses de validite.
    
    Les invariants se calculent a partir de SimulationResult, quelle que
    soit la representation de l'orientation.
"""
import numpy as np


def kinetic_energy(sphere, result):
    """
        T = 1/2 m |r_dot|^2 + 1/2 omega^T I_b omega (conservee), pour chaque k,
        avec r_dot = a (Omega_y, -Omega_x) et omega = R^T Omega
    """
    Omega = result.Omega
    r_dot = sphere.a * np.stack([Omega[:, 1], -Omega[:, 0]], axis=1)
    omega = np.einsum("kji,kj->ki", result.R, Omega)
    return (0.5 * sphere.m * np.sum(r_dot**2, axis=1)
            + 0.5 * np.einsum("ki,ij,kj->k", omega, sphere.Ib, omega))


def angular_momentum_P(sphere, result):
    """
        L_P = I_P Omega, moment cinetique au point de contact dans le repere
        sol (conserve comme vecteur), pour chaque k
    """
    return np.array([sphere.inertia_P(R) @ Om
                     for R, Om in zip(result.R, result.Omega)])


def no_slip_ok(sphere, friction):
    """
        Non-glissement : |(N_x, N_y)| <= mu_s m g
    """
    return np.linalg.norm(friction) <= sphere.mu_s * sphere.m * sphere.g


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
