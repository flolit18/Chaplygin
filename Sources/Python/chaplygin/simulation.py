"""
    Algorithme de simulation (partie II du chapitre 4).

        Etat Y = (alpha, alpha_dot) dans R^6.
        Etape 1 : RK4 sur Y_dot = f(Y).
        Etape 2 : alpha_ddot au nouveau pas.
        Etape 3 : r par la quadrature RK4 des pentes g_i = -A_alpha alpha_dot,
                lambda calcule directement.
"""
from dataclasses import dataclass
import numpy as np
import numpy as np

from .physics import E_space
from .controls import no_slip_ok, euler_ok


@dataclass
class SimulationResult:
    """Sorties de la simulation aux instants t_k = k h."""
    t: np.ndarray            # (N+1,)
    r: np.ndarray            # (N+1, 2)
    alpha: np.ndarray        # (N+1, 3)
    alpha_dot: np.ndarray    # (N+1, 3)
    lam: np.ndarray          # (N+1, 2)
    valid: bool = True       # False si un controle a echoue
    stop_reason: str = ""    # quel controle a echoue, et a quel pas


def f(sphere, Y):
    """
        Second membre Y_dot = (alpha_dot, alpha_ddot)
    """
    alpha, alpha_dot = Y[:3], Y[3:]
    return np.concatenate([alpha_dot, sphere.alpha_ddot(alpha, alpha_dot)])

def rk4_step(sphere, Y, h):
    """
        Un pas de RK4.

        Retours:
        
        Y_{k+1}
        dr : l'increment de r (avec les poids 1, 2, 2, 1 sur les g_i) et les etats intermediaires Y^(i) (pour tester cos(theta))
    """
    # Etape 1 : pentes p_i aux etats intermediaires Y^(i)
    Y1 = Y
    p1 = f(sphere, Y1)
    Y2 = Y + h/2 * p1
    p2 = f(sphere, Y2)
    Y3 = Y + h/2 * p2
    p3 = f(sphere, Y3)
    Y4 = Y + h * p3
    p4 = f(sphere, Y4)

    Y_next = Y + h/6 * (p1 + 2*p2 + 2*p3 + p4)

    # Etape 3 : r par les memes poids, g_i = r_dot(Y^(i)) = -A_alpha alpha_dot
    states = [Y1, Y2, Y3, Y4]
    g = [sphere.r_dot(Yi[:3], Yi[3:]) for Yi in states]
    dr = h/6 * (g[0] + 2*g[1] + 2*g[2] + g[3])

    return Y_next, dr, states


def initial_state(sphere, alpha0, Omega0):
    """
        Y_0 = (alpha0, alpha_dot0) avec alpha_dot0 = E_s(alpha0)^-1 Omega0
    """
    alpha0 = np.asarray(alpha0, dtype=float)
    Omega0 = np.asarray(Omega0, dtype=float)
    alpha_dot0 = np.linalg.solve(E_space(alpha0), Omega0)
    return np.concatenate([alpha0, alpha_dot0])


def simulate(sphere, r0, alpha0, Omega0, h, N, eps=1e-3, check=True):
    """
        Boucle sur k = 0, ..., N-1 : s'arrete si un controle echoue
    """
    Y = initial_state(sphere, alpha0, Omega0)
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
        lam = sphere.reaction(alpha, alpha_dot, sphere.alpha_ddot(alpha, alpha_dot))
        rs[k], Ys[k], lams[k] = r, Y, lam

        if check and not no_slip_ok(sphere, lam):
            valid, stop_reason = False, f"glissement au pas k = {k}"
            break
        if k == N:
            break

        # Etapes 1 et 3 : Y_{k+1} et r_{k+1}
        Y_next, dr, states = rk4_step(sphere, Y, h)
        if check and not euler_ok([Yi[:3] for Yi in states] + [Y_next[:3]], eps):
            valid, stop_reason = False, f"cos(theta) proche de 0 au pas k = {k}"
            break
        Y, r = Y_next, r + dr
        k += 1

    n = k + 1   # nombre d'instants effectivement calcules
    return SimulationResult(t=t[:n], 
                            r=rs[:n], 
                            alpha=Ys[:n, :3],
                            alpha_dot=Ys[:n, 3:], 
                            lam=lams[:n],
                            valid=valid, 
                            stop_reason=stop_reason)
