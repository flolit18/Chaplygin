"""
    Modele de la sphere de Chaplygin : parametres physiques communs aux
    deux representations de l'orientation (angles d'Euler et quaternions).
"""
import numpy as np


class ChaplyginSphere:
    """
        Sphere de rayon a, de masse m, centre de masse au centre geometrique,
        matrice d'inertie I_b (repere corps) quelconque, qui roule sans glisser
        sur un plan horizontal.

        Parametres
        ----------
        m : masse (kg)
        a : rayon (m)
        Ib : matrice d'inertie au centre dans le repere corps (3x3, kg m^2),
            ou vecteur des 3 moments principaux
        mu_s : coefficient de frottement statique (test de non-glissement)
        g : gravite (m/s^2)
        physics : solveur de l'orientation, "euler" ou "quaternion"
    """

    PHYSICS = ("euler", "quaternion")

    def __init__(self, m, a, Ib, mu_s=0.5, g=9.81, physics="quaternion"):
        self.m = float(m)
        self.a = float(a)
        Ib = np.asarray(Ib, dtype=float)
        self.Ib = np.diag(Ib) if Ib.ndim == 1 else Ib
        self.mu_s = float(mu_s)
        self.g = float(g)
        self.physics = physics

    def __repr__(self):
        return (f"ChaplyginSphere(m={self.m}, a={self.a}, "
                f"Ib=diag{np.diag(self.Ib).tolist()}, mu_s={self.mu_s}, "
                f"physics={self.physics!r})")

    @property
    def physics(self):
        """Solveur de l'orientation : "euler" ou "quaternion"."""
        return self._physics

    @physics.setter
    def physics(self, name):
        if name not in self.PHYSICS:
            raise ValueError(f"physics doit etre dans {self.PHYSICS}, pas {name!r}")
        self._physics = name

    # ------------------------------------------------------------------
    # Grandeurs conservees, pour un etat (R, Omega) ou une suite d'etats
    # faut noter que R : orientation et pas position
    #       (R de forme (N, 3, 3), Omega de forme (N, 3))
    # ------------------------------------------------------------------

    def kinetic_energy(self, R, Omega):
        """
            T = 1/2 m |v_G|^2 + 1/2 omega^T I_b omega,
            avec v_G = a (Omega_y, -Omega_x, 0) et omega = R^T Omega
        """
        R, Omega = np.asarray(R, dtype=float), np.asarray(Omega, dtype=float)
        v_G2 = self.a**2 * (Omega[..., 0]**2 + Omega[..., 1]**2)
        omega = np.einsum("...ji,...j->...i", R, Omega)
        return (0.5 * self.m * v_G2
                + 0.5 * np.einsum("...i,ij,...j->...", omega, self.Ib, omega))

    def angular_momentum_G(self, R, Omega):
        """
            L_G = R I_b R^T Omega, moment cinetique en G dans le repere sol
        """
        R, Omega = np.asarray(R, dtype=float), np.asarray(Omega, dtype=float)
        omega = np.einsum("...ji,...j->...i", R, Omega)
        return np.einsum("...ij,jk,...k->...i", R, self.Ib, omega)

    def angular_momentum_P(self, R, Omega):
        """
            L_P = I_P Omega = L_G + m a^2 (Omega_x, Omega_y, 0), moment
            cinetique au point de contact dans le repere sol (conserve)
        """
        Omega = np.asarray(Omega, dtype=float)
        L_P = self.angular_momentum_G(R, Omega).copy()
        L_P[..., :2] += self.m * self.a**2 * Omega[..., :2]
        return L_P

    def angular_momentum_Gz(self, R, Omega):
        """
            L_G . e_z, moment cinetique vertical (conserve : psi cyclique)
        """
        return self.angular_momentum_G(R, Omega)[..., 2]

    def simulate(self, r0, alpha0, Omega0, h, N, **options):
        """
            Simule N pas de RK4 de pas h avec le solveur choisi par physics
            Renvoie un SimulationResult.
        """
        # import local : euler/ et quaternion/ importent deja ce module
        if self.physics == "euler":
            from .euler.simulation import simulate
        else:
            from .quaternion.simulation import simulate
        return simulate(self, r0, alpha0, Omega0, h, N,**options)

    def inertia_P(self, R):
        """
            Inertie au point de contact dans le repere sol :
            I_P = R I_b R^T + m a^2 (Id - e_z e_z^T)
        """
        return R @ self.Ib @ R.T + self.m * self.a**2 * np.diag([1.0, 1.0, 0.0])
