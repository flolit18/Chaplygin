"""
    Dynamique de la sphere de Chaplygin en angles d'Euler.

    Coordonnees : r = (x, y) position du centre, alpha = (phi, theta, psi)
    angles d'Euler en convention Z-Y'-X'', R = Rz(psi) Ry(theta) Rx(phi).
    Le modele n'est valide que si cos(theta) != 0.
"""
import numpy as np

from ..rotations import E_body, E_dot_alpha_dot


# --- contrainte de roulement sans glissement : r_dot = -A_alpha alpha_dot

def A_alpha(sphere, alpha):
    """
        Matrice de contrainte A_alpha (2x3)
    """
    r, p, y = alpha

    cp = np.cos(p)
    cy, sy = np.cos(y), np.sin(y)

    return sphere.a * np.array([
        [-sy*cp, -cy, 0],
        [ cy*cp, -sy, 0],
    ])


def A_dot_alpha_dot(sphere, alpha, alpha_dot):
    """
        Produit Adot_alpha alpha_dot (2,)
    """
    r, p, y = alpha
    r_d, p_d, y_d = alpha_dot

    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return sphere.a * np.array([
        -r_d*y_d*cy*cp + r_d*p_d*sy*sp + p_d*y_d*sy,
        -r_d*y_d*sy*cp - r_d*p_d*cy*sp - p_d*y_d*cy,
    ])


# --- dynamique en orientation

def mass_matrix(sphere, alpha):
    """
        M_alpha = E^T I_b E
    """
    E = E_body(alpha)
    return E.T @ sphere.Ib @ E


def quadratic_terms(sphere, alpha, alpha_dot):
    """
        h_alpha = E^T (I_b Edot alpha_dot + omega x (I_b omega))
    """
    E = E_body(alpha)
    omega = E @ alpha_dot
    return E.T @ (sphere.Ib @ E_dot_alpha_dot(alpha, alpha_dot) + np.cross(omega, sphere.Ib @ omega))


def reduced_matrix(sphere, alpha):
    """
        K_alpha = M_alpha + m A_alpha^T A_alpha = E_s^T I_P E_s
    """
    A = A_alpha(sphere, alpha)
    return mass_matrix(sphere, alpha) + sphere.m * A.T @ A


def alpha_ddot(sphere, alpha, alpha_dot):
    """
        Resout le systeme reduit pour faire sortir alpha_ddot
        K_alpha alpha_ddot = -h_alpha - m A_alpha^T Adot_alpha alpha_dot
    """
    K = reduced_matrix(sphere, alpha)
    A = A_alpha(sphere, alpha)
    b = (-quadratic_terms(sphere, alpha, alpha_dot)
         - sphere.m * A.T @ A_dot_alpha_dot(sphere, alpha, alpha_dot))
    return np.linalg.solve(K, b)


def f(sphere, Y):
    """
        Second membre Y_dot = (alpha_dot, alpha_ddot), avec Y = (alpha, alpha_dot)
    """
    alpha, alpha_dot = Y[:3], Y[3:]
    return np.concatenate([alpha_dot, alpha_ddot(sphere, alpha, alpha_dot)])


# --- position et reactions

def r_dot(sphere, alpha, alpha_dot):
    """
        r_dot = -A_alpha alpha_dot
    """
    return -A_alpha(sphere, alpha) @ alpha_dot


def reaction(sphere, alpha, alpha_dot, alpha_ddot):
    """
        lambda = m r_ddot, 
            avec
        r_ddot = -(A_alpha alpha_ddot + Adot_alpha alpha_dot)
    """
    r_ddot = -(A_alpha(sphere, alpha) @ alpha_ddot
               + A_dot_alpha_dot(sphere, alpha, alpha_dot))
    return sphere.m * r_ddot
