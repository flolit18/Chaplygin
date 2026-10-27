"""
    Algorithme de simulation en angles d'Euler (partie II du chapitre 4).

        Etat Y = (alpha, alpha_dot) dans R^6.
        Etape 1 : RK4 sur Y_dot = f(Y).
        Etape 2 : alpha_ddot au nouveau pas.
        Etape 3 : r par la quadrature RK4 des pentes g_i = -A_alpha alpha_dot,
                lambda calcule directement.
"""
import numpy as np

from ..rotations import rotation_matrix, E_space
from ..rk4 import rk4_step
from ..result import SimulationResult
from ..controls import no_slip_ok, euler_ok, conserved_quantities_ok
from . import dynamics


def initial_state(sphere, alpha0, Omega0):
    """
        Y_0 = (alpha0, alpha_dot0) avec alpha_dot0 = E_s(alpha0)^-1 Omega0
    """
    alpha0 = np.asarray(alpha0, dtype=float)
    Omega0 = np.asarray(Omega0, dtype=float)
    alpha_dot0 = np.linalg.solve(E_space(alpha0), Omega0)
    
    R0 = rotation_matrix(alpha0)
    T0 = sphere.kinetic_energy(R0, Omega0)
    L_G0 = sphere.angular_momentum_G(R0, Omega0)
    return np.concatenate([alpha0, alpha_dot0]), T0, L_G0[2], np.linalg.norm(L_G0)



def simulate(sphere, r0, alpha0, Omega0, h, N, eps=1e-3, eps_schema = 1e-6, check=True):
    """
        Boucle sur k = 0, ..., N-1 : s'arrete si un controle echoue
    """
    f = lambda Y: dynamics.f(sphere, Y)
    g = lambda Y: dynamics.r_dot(sphere, Y[:3], Y[3:])

    Y, T0, Lz0, L0 = initial_state(sphere, alpha0, Omega0)
    r = np.asarray(r0, dtype=float)

    # tableaux de sortie, remplis au fur et a mesure
    t = h * np.arange(N + 1)
    rs = np.zeros((N + 1, 2))
    Ys = np.zeros((N + 1, 6))
    lams = np.zeros((N + 1, 2))

    valid, stop_reason = True, ""
    k = 0
    while True:
        # Etape 2 : alpha_ddot et lambda a l'instant t_k
        alpha, alpha_dot = Y[:3], Y[3:]
        alpha_ddot = dynamics.alpha_ddot(sphere, alpha, alpha_dot)
        lam = dynamics.reaction(sphere, alpha, alpha_dot, alpha_ddot)
        rs[k], Ys[k], lams[k] = r, Y, lam

        if check and not no_slip_ok(sphere, lam):
            valid, stop_reason = False, f"glissement au pas k = {k}"
            break
        R = rotation_matrix(alpha)
        if check and not conserved_quantities_ok(sphere, R, E_space(alpha) @ alpha_dot,
                                                 T0, Lz0, L0, eps_schema):
            valid, stop_reason = False, f"T ou L_G.e_z non conserve au pas k = {k}"
            break
        if k == N:
            break

        # Etapes 1 et 3 : Y_{k+1} et r_{k+1}
        Y_next, dr, states = rk4_step(f, Y, h, g)
        if check and not euler_ok([Yi[:3] for Yi in states] + [Y_next[:3]], eps):
            valid, stop_reason = False, f"cos(theta) proche de 0 au pas k = {k}"
            break
        Y, r = Y_next, r + dr
        k += 1

    n = k + 1   # nombre d'instants effectivement calcules
    alphas, alpha_dots = Ys[:n, :3], Ys[:n, 3:]
    return SimulationResult(t=t[:n],
                            r=rs[:n],
                            R=np.array([rotation_matrix(al) for al in alphas]),
                            Omega=np.array([E_space(al) @ ad
                                            for al, ad in zip(alphas, alpha_dots)]),
                            friction=lams[:n],
                            alpha=alphas,
                            valid=valid,
                            stop_reason=stop_reason)
