"""Capteurs simules, embarques dans la boule (repere corps).

Ils ne lisent que la sortie de la simulation (SimulationResult).
"""
import numpy as np


class Gyroscope:
    """Mesure omega (repere corps) avec un biais et un bruit blanc gaussien."""

    def __init__(self, noise_std=0.0, bias=(0.0, 0.0, 0.0), seed=None):
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float)
        self.rng = np.random.default_rng(seed)

    def measure(self, sphere, result):
        """Retourne les mesures (N+1, 3)."""
        raise NotImplementedError


class Accelerometer:
    """Mesure la force specifique au centre, dans le repere corps :
    R^T (r_ddot_3D - g_vec), avec un bruit blanc gaussien."""

    def __init__(self, noise_std=0.0, bias=(0.0, 0.0, 0.0), seed=None):
        self.noise_std = noise_std
        self.bias = np.asarray(bias, dtype=float)
        self.rng = np.random.default_rng(seed)

    def measure(self, sphere, result):
        """Retourne les mesures (N+1, 3)."""
        raise NotImplementedError
