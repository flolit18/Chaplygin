"""
    Algorithme de simulation en quaternions (partie III.3 du chapitre 4).

        Etat Y = (q, omega) dans R^7.
        Etape 1 : RK4 sur Y_dot = f(Y).
        Etape 2 : renormalisation q <- q / |q|.
        Etape 3 : omega_dot au nouveau pas.
        Etape 4 : r par la quadrature RK4 des pentes g_i = a (Omega_y, -Omega_x),
                frottements calcules directement.
"""
import numpy as np

from ..rotations import quat_from_euler, quat_to_matrix, euler_from_matrix
from ..rk4 import rk4_step
from ..result import SimulationResult
from ..controls import no_slip_ok
from . import dynamics


def initial_state(alpha0, Omega0):
    """
        Y_0 = (q(alpha0), omega0) avec omega0 = R(q(alpha0))^T Omega0.
        Aucune condition sur cos(theta0)
    """
    q0 = quat_from_euler(alpha0)
    omega0 = quat_to_matrix(q0).T @ np.asarray(Omega0, dtype=float)
    return np.concatenate([q0, omega0])


def simulate(sphere, r0, alpha0, Omega0, h, N, check=True):
    """
        Boucle sur k = 0, ..., N-1 : s'arrete si le non-glissement echoue
    """
    f = lambda Y: dynamics.f(sphere, Y)
    g = lambda Y: dynamics.r_dot(sphere, Y[:4], Y[4:])

    Y = initial_state(alpha0, Omega0)
    r = np.asarray(r0, dtype=float)

    # tableaux de sortie, remplis au fur et a mesure
    t = h * np.arange(N + 1)
    rs = np.zeros((N + 1, 2))
    Ys = np.zeros((N + 1, 7))
    frics = np.zeros((N + 1, 2))

    valid, stop_reason = True, ""
    k = 0
    while True:
        # Etape 3 : omega_dot et frottements a l'instant t_k
        q, omega = Y[:4], Y[4:]
        omega_dot = dynamics.omega_dot(sphere, q, omega)
        fric = dynamics.friction(sphere, q, omega_dot)
        rs[k], Ys[k], frics[k] = r, Y, fric

        if check and not no_slip_ok(sphere, fric):
            valid, stop_reason = False, f"glissement au pas k = {k}"
            break
        if k == N:
            break

        # Etapes 1 et 4 : Y_{k+1} et r_{k+1}
        Y_next, dr, _ = rk4_step(f, Y, h, g)

        # Etape 2 : renormalisation
        Y_next[:4] /= np.linalg.norm(Y_next[:4])
        Y, r = Y_next, r + dr
        k += 1

    n = k + 1   # nombre d'instants effectivement calcules
    qs, omegas = Ys[:n, :4], Ys[:n, 4:]
    Rs = np.array([quat_to_matrix(q) for q in qs])

    # angles d'Euler pour l'affichage, sans les sauts de 2 pi
    alphas = np.array([euler_from_matrix(R) for R in Rs])
    alphas[:, 0] = np.unwrap(alphas[:, 0])
    alphas[:, 2] = np.unwrap(alphas[:, 2])

    return SimulationResult(t=t[:n],
                            r=rs[:n],
                            R=Rs,
                            Omega=np.einsum("kij,kj->ki", Rs, omegas),
                            friction=frics[:n],
                            alpha=alphas,
                            q=qs,
                            valid=valid,
                            stop_reason=stop_reason)
