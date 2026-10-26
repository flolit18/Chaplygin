"""
    Modele de la sphere de Chaplygin.

    Coordonnees : r = (x, y) position du centre, alpha = (phi, theta, psi)
    angles d'Euler en convention Z-Y'-X'', R = Rz(psi) Ry(theta) Rx(phi).
    Le modele n'est valide que si cos(theta) != 0.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Les matrices necessaires pour la simulation
# ---------------------------------------------------------------------------

def rotation_matrix(alpha):
    """
        Matrice de rotation R = Rz(psi) Ry(theta) Rx(phi), repere corps -> sol
    """
    r, p, y = alpha
    
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    


def E_body(alpha):
    """
        Matrice E telle que omega = E alpha_dot 
        (vitesse angulaire dans lerepere corps) et det E = cos(theta)
    """
    r, p, y = alpha
    
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    
    return np.array([
        [1,   0,  -sp],
        [0,  cr, sr*cp],
        [0, -sr, cr*cp],
    ])

def E_space(alpha):
    """
        Matrice E_s telle que Omega = E_s alpha_dot
        (vitesse angulaire dans le repere sol) et E_s = R E, det E_s = cos(theta)
    """
    r, p, y = alpha

    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        [cy*cp, -sy, 0],
        [sy*cp,  cy, 0],
        [  -sp,   0, 1],
    ])


def E_dot_alpha_dot(alpha, alpha_dot):
    """
        Produit Edot alpha_dot, d/dt(E(alpha))
        
        Edot ijk = dE_ij/dalpha_k alpha_dot_k
        
        E ne depend que de phi et theta :
        Edot = dE/dphi phi_dot + dE/dtheta theta_dot
    """
    r, p, y = alpha
    r_d, p_d, y_d = alpha_dot

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)

    return np.array([
        -p_d*y_d*cp,
        -r_d*p_d*sr + y_d*(r_d*cr*cp - p_d*sr*sp),
        -r_d*p_d*cr - y_d*(r_d*sr*cp + p_d*cr*sp),
    ])


# ---------------------------------------------------------------------------
# La boule
# ---------------------------------------------------------------------------

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
    """

    def __init__(self, m, a, Ib, mu_s=0.5, g=9.81):
        self.m = float(m)
        self.a = float(a)
        Ib = np.asarray(Ib, dtype=float)
        self.Ib = np.diag(Ib) if Ib.ndim == 1 else Ib
        self.mu_s = float(mu_s)
        self.g = float(g)

    def __repr__(self):
        return (f"ChaplyginSphere(m={self.m}, a={self.a}, "
                f"Ib=diag{np.diag(self.Ib).tolist()}, mu_s={self.mu_s})")

    # --- contrainte de roulement sans glissement : r_dot = -A_alpha alpha_dot

    def A_alpha(self, alpha):
        """
            Matrice de contrainte A_alpha (2x3)
        """
        r, p, y = alpha

        cp = np.cos(p)
        cy, sy = np.cos(y), np.sin(y)

        return self.a * np.array([
            [-sy*cp, -cy, 0],
            [ cy*cp, -sy, 0],
        ])

    def A_dot_alpha_dot(self, alpha, alpha_dot):
        """
            Produit Adot_alpha alpha_dot (2,)
        """
        r, p, y = alpha
        r_d, p_d, y_d = alpha_dot

        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)

        return self.a * np.array([
            -r_d*y_d*cy*cp + r_d*p_d*sy*sp + p_d*y_d*sy,
            -r_d*y_d*sy*cp - r_d*p_d*cy*sp - p_d*y_d*cy,
        ])

    # --- dynamique en orientation

    def mass_matrix(self, alpha):
        """
            M_alpha = E^T I_b 
        """
        E = E_body(alpha)
        return E.T @ self.Ib @ E

    def quadratic_terms(self, alpha, alpha_dot):
        """
            h_alpha = E^T (I_b Edot alpha_dot + omega x (I_b omega))
        """
        E = E_body(alpha)
        omega = E @ alpha_dot
        return E.T @ (self.Ib @ E_dot_alpha_dot(alpha, alpha_dot) + np.cross(omega, self.Ib @ omega))

    def reduced_matrix(self, alpha):
        """
            K_alpha = M_alpha + m A_alpha^T A_alpha = E_s^T I_P E_s
        """
        A = self.A_alpha(alpha)
        return self.mass_matrix(alpha) + self.m * A.T @ A

    def inertia_P(self, alpha):
        """
            Inertie au point de contact dans le repere sol :
            I_P = R I_b R^T + m a^2 (Id - e_z e_z^T)
        """
        R = rotation_matrix(alpha)
        return R @ self.Ib @ R.T + self.m * self.a**2 * np.diag([1.0, 1.0, 0.0])

    def alpha_ddot(self, alpha, alpha_dot):
        """
            Resout le systeme reduit pour faire sortir alpha_ddot
            K_alpha ---alpha_ddot--- = -h_alpha - m A_alpha^T Adot_alpha alpha_dot
        """
        K = self.reduced_matrix(alpha)
        A = self.A_alpha(alpha)
        b = (-self.quadratic_terms(alpha, alpha_dot)
             - self.m * A.T @ self.A_dot_alpha_dot(alpha, alpha_dot))
        return np.linalg.solve(K, b)

    # --- position et reactions

    def r_dot(self, alpha, alpha_dot):
        """
            r_dot = -A_alpha alpha_dot
        """
        return -self.A_alpha(alpha) @ alpha_dot

    def reaction(self, alpha, alpha_dot, alpha_ddot):
        """
            lambda = m r_ddot, 
                avec
            r_ddot = -(A_alpha alpha_ddot + Adot_alpha alpha_dot)
        """
        r_ddot = -(self.A_alpha(alpha) @ alpha_ddot
                   + self.A_dot_alpha_dot(alpha, alpha_dot))
        return self.m * r_ddot
