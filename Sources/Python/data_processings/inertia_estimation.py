"""
    Traitement des donnees du gyrometre pour estimer la matrice d'inertie.

    Dans le repere du gyro, J est symetrique (pas diagonale) et les deux
    quantites conservees sont lineaires en
        theta = [J11 J22 J33 J12 J23 J13 c T]^T   dans R^8
    avec theta le vecteur des params a estimer 
    
    avec c = L_G . e_z et T l'energie cinetique :
    
        varphi_1(gamma, omega)^T theta = 0          (L_G . e_z = c)
        varphi_2(gamma, omega)^T theta = y_2        (energie = T)
        
    En empilant les deux lignes pour chaque mesure k :
        Phi theta = y (big Phi)

    A lancer depuis le dossier Python 

    python -m data_processings.inertia_estimation
"""
import csv
from pathlib import Path

import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin.rotations import quat_from_euler, quat_to_matrix

# ordre des parametres dans theta
THETA_NAMES = ["J11", "J22", "J33", "J12", "J23", "J13", "c", "T"]


#----------------------------------------------------------------------------------------------------------------------#
# Lecture des donnees, il faut lancer la simulation (collectdata)
#----------------------------------------------------------------------------------------------------------------------#
def load_gyro_csv(path):
    """
        Lit le csv du gyro : colonnes t, omega_x, omega_y, omega_z
        
        Retourne t (n,) en s et omega (n, 3) en rad/s, repere du capteur
    """
    with Path(path).open(newline = "", encoding = "utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(row["t"]) for row in rows])
    omega = np.array([[float(row["omega_x"]), float(row["omega_y"]), float(row["omega_z"])]
                      for row in rows])
    return t, omega


#----------------------------------------------------------------------------------------------------------------------#
# Verticale gamma = R^T e_z dans le repere du capteur, 
#----------------------------------------------------------------------------------------------------------------------#
def rodrigues(v):
    """
        exp([v]x) : rotation d'angle |v| autour de v / |v|
    """
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def integrate_gamma(t, omega, gamma0):
    """
        Integre gamma_dot = gamma ^ omega = -[omega]x gamma a partir de gamma0.
        
        Sur chaque pas, on fait la rotation exacte avec omega moyen :
            gamma_{k+1} = exp(-[omega_moy]x dt) gamma_k
            
        ce qui garde |gamma| = 1
        
        Le biais du gyro s'accumule dans gamma (c'est un probleme qu'il faudra effectivement regler
        mais pr l'instnat on suppose que le biais est petit et que cela ne drift pas)
    """
    gamma = np.zeros_like(omega)
    gamma[0] = gamma0 / np.linalg.norm(gamma0)
    for k in range(len(t) - 1):
        dt = t[k+1] - t[k]
        omega_moy = 0.5 * (omega[k] + omega[k+1])
        gamma[k+1] = rodrigues(-omega_moy * dt) @ gamma[k]
    return gamma


#----------------------------------------------------------------------------------------------------------------------#
# Regresseurs
#----------------------------------------------------------------------------------------------------------------------#
def varphi_1(gamma, omega):
    """
        L_G . e_z = gamma^T J omega = c  -->  varphi_1^T theta = 0
        Retourne (n, 8), une ligne par mesure
    """
    g1, g2, g3 = gamma.T
    w1, w2, w3 = omega.T
    zeros, ones = np.zeros_like(w1), np.ones_like(w1)
    return np.column_stack([g1*w1, g2*w2, g3*w3,
                            g1*w2 + g2*w1,      # J12
                            g2*w3 + g3*w2,      # J23
                            g1*w3 + g3*w1,      # J13
                            -ones,              # c
                            zeros])             # T


def varphi_2(omega):
    """
        2T = omega^T J omega + m a^2 (|omega|^2 - (gamma . omega)^2)
        --> phi_2^T theta = y_2
        Retourne (n, 8), une ligne par mesure
    """
    w1, w2, w3 = omega.T
    zeros, ones = np.zeros_like(w1), np.ones_like(w1)
    return np.column_stack([w1**2, w2**2, w3**2,
                            2*w1*w2,            # J12
                            2*w2*w3,            # J23
                            2*w1*w3,            # J13
                            zeros,              # c
                            -2*ones])           # T


def y_2(m, a, gamma, omega):
    """
        y_2 = -m a^2 |v_G / a|^2 = -m a^2 (|omega|^2 - (gamma . omega)^2)
    """
    return -m * a**2 * (np.sum(omega**2, axis=1) - np.sum(gamma * omega, axis=1)**2)


def big_phi(m, a, gamma, omega):
    """
        Empile les deux lignes de chaque mesure k :
        
            Phi_k = [phi_1^T ; phi_2^T]  (2, 8),   y_k = [0 ; y_2]
            
        Retourne Phi (n, 2, 8) et y (n, 2) ; Phi.reshape(-1, 8) donne le
        systeme complet (2n, 8) pour les moindres carres
    """
    
    Phi = np.stack([varphi_1(gamma, omega), varphi_2(omega)], axis=1)
    y = np.column_stack([np.zeros(len(omega)), y_2(m, a, gamma, omega)])
    return Phi, y


#----------------------------------------------------------------------------------------------------------------------#
# Vecteur des parametres
#----------------------------------------------------------------------------------------------------------------------#
def theta_from_params(J, c, T):
    """
        theta = [J11 J22 J33 J12 J23 J13 c T]
    """
    return np.array([J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[1, 2], J[0, 2], c, T])


def params_from_theta(theta):
    """
        Inverse de theta_from_params : retourne J (3, 3) symetrique, c, T
    """
    J11, J22, J33, J12, J23, J13, c, T = theta
    J = np.array([[J11, J12, J13],
                  [J12, J22, J23],
                  [J13, J23, J33]])
    return J, c, T


def estimate_theta(Phi, y):
    """
        loss function : 1/2 ||y - Phi theta||^2
        
        Moindres carres : theta_est = argmin 1/2 ||y - Phi theta||^2
        
        Equations normales (gradient nul) :
            Phi^T Phi theta = Phi^T y  -->  theta_est = (Phi^T Phi)^-1 Phi^T y
            
        On resout le systeme 8 x 8 au lieu de calculer l'inverse (plus stable), 
        tant que c'est simple on le garde, si jamais y a des problem, on 
        passe a un truc enligne
    """
    A = Phi.reshape(-1, 8)      # (2n, 8) : les deux lignes de chaque mesure
    b = y.ravel()               # (2n,)
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    # boule et lancement : memes params que simulate_sphere_collectdata
    m = 1.0 #kg
    a = 0.1 #m
    Ib = np.array([1.5e-3, 8.5e-3, 9.5e-3])
    alpha0 = np.deg2rad([10.0, 20.0, 10.0])
    Omega0 = np.array([0.8, 0.10, 5.30]) # rad/s

    # donnees du gyro
    t, omega = load_gyro_csv(Path("synthetic_data") / "gyro.csv")
    print("mesures :", len(t), "de t =", t[0], "a", t[-1], "s")

    # gamma0 : verticale initiale dans le repere du capteur, ici connue par la simulation
    # (en vrai, on suppose avoir un accelerometre, donc gamma0 est donnee par l'accelerometre si la boule part au repos)
    R0 = quat_to_matrix(quat_from_euler(alpha0))
    gamma0 = R0.T @ np.array([0.0, 0.0, 1.0])
    gamma = integrate_gamma(t, omega, gamma0)
    # la il faut integrer gamma, c'est pas un truc qu'on connait pdt la sim
    
    Phi, y = big_phi(m, a, gamma, omega)
    print("Phi :", Phi.shape, " y :", y.shape)

    # verification : avec les vrais params, le residu doit etre petit (bruit + biais du gyro)
    #sphere = ChaplyginSphere(m = m, a = a, Ib = Ib)
    #c0 = sphere.angular_momentum_Gz(R0, Omega0)
    #T0 = sphere.kinetic_energy(R0, Omega0)
    #theta_true = theta_from_params(np.diag(Ib), c0, T0)
    #residu = y - Phi @ theta_true
    #print("theta vrai :", dict(zip(THETA_NAMES, np.round(theta_true, 6))))
    #print("residu relatif L_G.e_z :", np.abs(residu[:, 0]).max() / abs(c0))
    #print("residu relatif energie :", np.abs(residu[:, 1]).max() / (2 * T0))
    """
    Je voulais faire une methode en ligne mais tant que c lineaire puis pour etre simple, on peut 
    utiliser directement la regression lineaire
    """
    theta_est = estimate_theta(Phi, y)
    J_est, c_est, T_est = params_from_theta(theta_est)
    print("theta estime :", dict(zip(THETA_NAMES, np.round(theta_est, 6))))

    # moments principaux = valeurs propres de J, axes principaux = vecteurs propres
    I_est, axes_est = np.linalg.eigh(J_est)
    print("moments principaux estimes :", I_est)
    print("l'erreur d'estimation :", np.linalg.norm(Ib - I_est))
    

if __name__ == "__main__":
    main()
