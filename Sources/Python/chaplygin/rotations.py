"""
    Representations de l'orientation.

    Angles d'Euler : alpha = (phi, theta, psi), convention Z-Y'-X'',
    R = Rz(psi) Ry(theta) Rx(phi), repere corps -> sol.
    Quaternions : q = (q0, q1, q2, q3), partie scalaire en premier,
    produit de Hamilton, R(q) repere corps -> sol (chapitre 2).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Angles d'Euler
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


def euler_from_matrix(R):
    """
        Angles d'Euler (phi, theta, psi) d'une matrice de rotation,
        avec theta dans [-pi/2, pi/2] (singulier si cos(theta) = 0)
    """
    theta = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------

def hat(v):
    """
        Matrice [v]_x telle que [v]_x u = v x u
    """
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def quat_mul(q, p):
    """
        Produit de Hamilton q * p = (q0 p0 - q.p, q0 p + p0 q + q x p)
    """
    q0, qv = q[0], q[1:]
    p0, pv = p[0], p[1:]
    return np.concatenate([[q0*p0 - qv @ pv], q0*pv + p0*qv + np.cross(qv, pv)])


def quat_to_matrix(q):
    """
        R(q) = (q0^2 - |q|^2) Id + 2 q q^T + 2 q0 [q]_x, repere corps -> sol.
        q est normalise avant le calcul (les etats intermediaires de RK4
        ne sont pas exactement unitaires)
    """
    # donc il faut renormaliser ici
    q = q / np.linalg.norm(q)
    q0, qv = q[0], q[1:]
    return (q0**2 - qv @ qv) * np.eye(3) + 2*np.outer(qv, qv) + 2*q0*hat(qv)


def quat_from_euler(alpha):
    """
        Quaternion des angles d'Euler (Z-Y'-X'') :
        q = q_z(psi) * q_y(theta) * q_x(phi)
    """
    r, p, y = np.asarray(alpha, dtype=float) / 2

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def euler_from_quat(q):
    """
        Angles d'Euler d'un quaternion, pour l'affichage seulement
        (singulier si cos(theta) = 0)
    """
    return euler_from_matrix(quat_to_matrix(q))
