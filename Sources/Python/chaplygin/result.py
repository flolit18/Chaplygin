"""
    Sorties communes des simulations (angles d'Euler ou quaternions).

    L'orientation est stockee par la matrice R et la vitesse angulaire par
    Omega (repere sol) : les controles et les figures ne dependent pas de la
    representation utilisee.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """
        Sorties de la simulation aux instants t_k = k h
    """
    t: np.ndarray            # (N+1,)
    r: np.ndarray            # (N+1, 2) position du centre
    R: np.ndarray            # (N+1, 3, 3) orientation, repere corps -> sol
    Omega: np.ndarray        # (N+1, 3) vitesse angulaire, repere sol
    friction: np.ndarray     # (N+1, 2) frottements (N_x, N_y) = lambda
    alpha: np.ndarray        # (N+1, 3) angles d'Euler, pour l'affichage
    q: np.ndarray = None     # (N+1, 4) quaternion, simulation en quaternions
    valid: bool = True       # False si un controle a echoue
    stop_reason: str = ""    # quel controle a echoue, et a quel pas
