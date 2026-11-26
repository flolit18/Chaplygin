"""
    Code pour simuler un accelerometre.

    Meme IMU que le gyrometre (MPU6050, 100 Hz), donc meme montage Q.
    Le capteur est suppose au centre de la boule.

    Force specifique vue depuis le corps (partie capteurs du chapitre 6) :
        f = g gamma + a omega_dot x gamma
"""
import numpy as np

from chaplygin.rotations import quat_to_matrix
from chaplygin.quaternion.dynamics import K


class Accelerometer:
    def __init__(self, sigma_a = 0.03, biais_std = 0.05, sigma_ba = 1e-3, freq = 100,
                 mode = "ideal", seed = 43, mount = None):
        """
            Mesure :
                z_k = g gamma_k + a omega_dot_k x gamma_k + b_k + nu_k

            b_k : biais a t = k, fixe ("ideal") ou marche aleatoire ("realist")
                        b_{k+1} = b_k + w_k,   w_k ~ N(0, sigma_ba^2 dt I_3)
            nu_k : bruit blanc,   nu_k ~ N(0, sigma_a^2 I_3)

            mount : rotation Q du repere capteur vers le repere corps,
                    v_capteur = Q^T v_corps. Passer gyro.mount (meme IMU).
                    Si None, Q tiree uniformement sur SO(3)
        """
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        # ecart type du bruit (m/s^2)
        self.sigma_a = sigma_a

        # biais initial et derive (m/s^2, m/s^2/sqrt(s))
        self.biais = self.rng.normal(0, biais_std, 3)
        self.sigma_ba = sigma_ba
        # frequence du capteur (Hz)
        self.freq = freq

        if mount is None:
            q = self.rng.normal(size = 4)
            mount = quat_to_matrix(q / np.linalg.norm(q))
        self.mount = np.asarray(mount, dtype = float)

    def mesure(self, result, sphere):
        """
            z_k = g gamma_k + a omega_dot_k x gamma_k + b_k + nu_k

        gamma, omega et omega_dot sont les vraies valeurs de la simulation,
        prises tous les tick = (1 / freq) / h pas
        """
        h = result.t[1] - result.t[0]
        ratio = (1.0 / self.freq) / h
        tick = round(ratio)
        if tick < 1 or abs(ratio - tick) > 1e-9:
            raise ValueError(f"h = {h} doit diviser la periode du capteur 1/freq = {1/self.freq}")

        R, Omega = result.R[::tick], result.Omega[::tick]
        # gamma = R^T e_z (3e ligne de R), omega = R^T Omega
        gamma = R[:, 2, :]
        omega = np.einsum("nji,nj->ni", R, Omega)
        # omega_dot : K(gamma) omega_dot = -omega x (I_b omega)
        rhs = -np.cross(omega, omega @ sphere.Ib.T)
        omega_dot = np.array([np.linalg.solve(K(sphere, gam), b) for gam, b in zip(gamma, rhs)])

        # force specifique dans le corps, puis dans le repere capteur
        f_body = sphere.g * gamma + sphere.a * np.cross(omega_dot, gamma)
        f = f_body @ self.mount

        noise = self.rng.normal(0, self.sigma_a, f.shape)

        if self.mode == "realist":
            # marche aleatoire : b_k = b_0 + sum_j w_j,  w_j ~ N(0, sigma_ba^2 dt)
            dt = 1.0 / self.freq
            w = self.rng.normal(0, self.sigma_ba * np.sqrt(dt), f.shape)
            w[0] = 0.0
            biais = self.biais + np.cumsum(w, axis = 0)
        else:
            biais = np.broadcast_to(self.biais, f.shape)

        # vraies valeurs dans le repere capteur, pour comparer avec l'EKF
        self.biais_traj = biais
        self.gamma_traj = gamma @ self.mount
        return result.t[::tick], f + biais + noise
