"""
    Filtre de Kalman etendu sur x = (gamma, b_g, b_a) (chapitre 6, partie III).

    Entree   : gyrometre      z_g = omega + b_g + nu_g
    Mesure   : accelerometre  z_a = g gamma + a omega_dot x gamma + b_a + nu_a

    Tout est exprime dans le repere capteur (meme IMU pour les deux capteurs) :
    les equations sont invariantes par une rotation fixe.
    
    Pas de latence, capteur de timing parfait.
"""
import numpy as np

from chaplygin.rotations import hat


def exp_so3(phi):
    """
        exp(hat(phi)) par la formule de Rodrigues
    """
    theta = np.linalg.norm(phi)
    S = hat(phi)
    if theta < 1e-12:
        return np.eye(3) + S
    return (np.eye(3) + np.sin(theta) / theta * S + (1 - np.cos(theta)) / theta**2 * S @ S)


class EKF:
    def __init__(self, a, g, dt, sigma_g, sigma_bg, sigma_a, sigma_ba, x0, P0):
        """
            a, g : rayon de la boule et gravite
            dt : periode des capteurs
            sigma_g : bruit du gyro,            nu_g ~ N(0, sigma_g^2 I)
            sigma_bg : derive du biais gyro,     w_g  ~ N(0, sigma_bg^2 dt I)
            sigma_a : bruit de l'accelerometre, nu_a ~ N(0, sigma_a^2 I)
            sigma_ba : derive du biais accelero, w_a  ~ N(0, sigma_ba^2 dt I)
            x0, P0 : etat initial (9,) et covariance initiale (9, 9)
        """
        self.a, self.g, self.dt = a, g, dt
        self.sigma_g, self.sigma_bg = sigma_g, sigma_bg
        self.sigma_a, self.sigma_ba = sigma_a, sigma_ba
        self.x = np.array(x0, dtype = float)
        self.P = np.array(P0, dtype = float)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, z_g):
        """
            gamma_{k+1|k} = E_k gamma_{k|k},  E_k = exp(-dt hat(z_g - b_g))
            P_{k+1|k}     = F_k P_{k|k} F_k^T + Q_k
        """
        dt = self.dt
        gam, b_g = self.x[:3], self.x[3:6]
        E = exp_so3(-dt * (z_g - b_g))

        # jacobienne de la dynamique
        F = np.eye(9)
        F[:3, :3] = E
        F[:3, 3:6] = -dt * hat(gam)

        # bruit de processus : Q = G E[n n^T] G^T
        G_gam = -dt * hat(gam)
        Q = np.zeros((9, 9))
        Q[:3, :3] = self.sigma_g**2 * G_gam @ G_gam.T
        Q[3:6, 3:6] = self.sigma_bg**2 * dt * np.eye(3)
        Q[6:, 6:] = self.sigma_ba**2 * dt * np.eye(3)

        # les biais sont constants dans F
        self.x[:3] = E @ gam
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def update(self, z_a, omega_dot_hat):
        """
            Correction par l'accelerometre, forme de Joseph,
            puis renormalisation de gamma. Renvoie la NIS y^T S^-1 y
        """
        gam = self.x[:3]

        # mesure lineaire : z_a = H x + nu_a_tilde
        H = np.zeros((3, 9))
        H[:, :3] = self.g * np.eye(3) + self.a * hat(omega_dot_hat)
        H[:, 6:] = np.eye(3)

        # R gonflee : erreur sur omega_dot par difference finie
        Hg = hat(gam) #asym gamma
        R = (self.sigma_a**2 * np.eye(3) + 2 * self.a**2 * self.sigma_g**2 / self.dt**2 * Hg @ Hg.T)

        # Innovation
        y = z_a - H @ self.x
        S = H @ self.P @ H.T + R
        
        # Gain
        K = np.linalg.solve(S, H @ self.P).T     # K = P H^T S^-1 (S, P symetriques)

        # Update etat
        self.x = self.x + K @ y
        # Renormalisation
        self.x[:3] /= np.linalg.norm(self.x[:3])

        # forme de Joseph
        IKH = np.eye(9) - K @ H
        # Update cov
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return y @ np.linalg.solve(S, y)

    # ------------------------------------------------------------------
    # Boucle complete
    # ------------------------------------------------------------------

    def run(self, z_g, z_a):
        """
            z_g, z_a : mesures (N, 3) aux memes instants t_k = k dt

            omega_dot_hat_k = (z_{g,k+1} - z_{g,k}) / dt  (derniere : difference arriere)
            Etape k -> k+1 : prediction avec z_{g,k}, correction avec z_{a,k+1}

            Renvoie xs (N, 9), Ps (N, 9, 9), nis (N,) (nis[0] = nan)
        """
        N = len(z_g)
        
        omega_dot_hat = np.empty_like(z_g)
        omega_dot_hat[:-1] = np.diff(z_g, axis = 0) / self.dt
        omega_dot_hat[-1] = omega_dot_hat[-2]

        xs = np.zeros((N, 9))
        Ps = np.zeros((N, 9, 9))
        nis = np.full(N, np.nan)
        xs[0], Ps[0] = self.x, self.P

        for k in range(N - 1):
            self.predict(z_g[k])
            nis[k + 1] = self.update(z_a[k + 1], omega_dot_hat[k + 1])
            xs[k + 1], Ps[k + 1] = self.x, self.P
            
        return xs, Ps, nis


def initial_state(z_a0, sigma_gam0 = 0.1, sigma_bg0 = 0.01, sigma_ba0 = 0.1):
    """
        gamma_0 = z_a0 / |z_a0| (accelero ~ g gamma au debut), biais nuls
    """
    x0 = np.zeros(9)
    x0[:3] = z_a0 / np.linalg.norm(z_a0)
    P0 = np.diag(np.concatenate([np.full(3, sigma_gam0**2),
                                 np.full(3, sigma_bg0**2),
                                 np.full(3, sigma_ba0**2)]))
    return x0, P0
