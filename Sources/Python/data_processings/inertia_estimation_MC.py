"""
    Estimation de l'inertie sur toute la campagne Monte Carlo 

    error relative pour la simulation k 
        eps_k = ||J_est - J|| _F / ||J|| _F
        
    avec J = Q^T I_b Q la vraie matrice d'inertie dans le repere du gyro

    Ensuite, pour chaque tolerance tol :
        S_k = 1{eps_k < tol}  ~ Bernoulli(p)
        p_est = moyenne des S_k,  IC 95 % : p_est +- 1.96 sqrt(p_est (1 - p_est) / N)

    Sorties :
    
        monte_carlo/estimation_results.csv : une ligne par simulation
        monte_carlo/hist_eps.png           : histogramme de eps

    A lancer depuis le dossier Python de la source :

    python -m data_processings.inertia_estimation_MC
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_processings.inertia_estimation import (integrate_gamma, big_phi,
                                                 estimate_theta, params_from_theta)

# tolerances sur eps pour le taux de reussite
TOLS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]


#----------------------------------------------------------------------------------------------------------------------#
# Lecture des donnees
#----------------------------------------------------------------------------------------------------------------------#
def load_campaign(folder):
    """
        Retourne :
            gyro   (n, 4 N) : t, omega_x, omega_y, omega_z de chaque simulation, cote a cote
            gamma0 (N, 3)
            mount  (N, 3, 3) : montage Q, repere du gyro -> repere corps
            truth  (N, 5) : m, a, I1, I2, I3
    """
    # la premiere colonne des 3 petits fichiers est le numero de simulation
    gamma0 = np.loadtxt(folder / "gamma0.csv", delimiter = ",", skiprows = 1)[:, 1:]
    mount = np.loadtxt(folder / "mount.csv", delimiter = ",", skiprows = 1)[:, 1:].reshape(-1, 3, 3)
    truth = np.loadtxt(folder / "truth.csv", delimiter = ",", skiprows = 1)[:, 1:6]
    print("lecture de gyro.csv (gros fichier)...")
    gyro = np.loadtxt(folder / "gyro.csv", delimiter = ",", skiprows = 1)
    return gyro, gamma0, mount, truth


#----------------------------------------------------------------------------------------------------------------------#
# Une simulation
#----------------------------------------------------------------------------------------------------------------------#
def analyse_simulation(t, omega, gamma0, Q, m, a, Ib):
    """
        Estime J dans le repere du gyro et la compare a la verite terrain.
        Retourne eps (Frobenius), l'erreur relative max sur les moments et
        l'erreur max sur les axes (deg). Si Phi^T Phi est singuliere : eps = inf
    """
    J_true = Q.T @ np.diag(Ib) @ Q

    gamma = integrate_gamma(t, omega, gamma0)
    Phi, y = big_phi(m, a, gamma, omega)
    
    try:
        J_est, c_est, T_est = params_from_theta(estimate_theta(Phi, y))
    except np.linalg.LinAlgError:
        return np.inf, np.inf, np.inf

    # erreur de Frobenius : ne depend pas du montage
    eps = np.linalg.norm(J_est - J_true) / np.linalg.norm(J_true)

    # moments et axes principaux, ranges par moments croissants
    I_est, axes_est = np.linalg.eigh(J_est)
    order = np.argsort(Ib)
    I_true, axes_true = Ib[order], Q.T[:, order]    # axes principaux vus du gyro = colonnes de Q^T
    err_I = np.max(np.abs(I_est - I_true) / I_true)
    
    # angle entre axe estime et axe vrai, au signe pres
    # attention : n'a pas de sens si deux moments sont presque egaux
    cos_axes = np.abs(np.sum(axes_est * axes_true, axis = 0))
    err_axes = np.max(np.degrees(np.arccos(np.clip(cos_axes, 0, 1))))
    return eps, err_I, err_axes


#----------------------------------------------------------------------------------------------------------------------#
# Taux de reussite
#----------------------------------------------------------------------------------------------------------------------#
def success_rate(eps, tol):
    """
        S_k = 1{eps_k < tol},  p_est = moyenne des S_k
        IC 95 % (TCL) : p_est +- 1.96 sqrt(p_est (1 - p_est) / N)
    """
    S = eps < tol
    N = len(S)
    p_est = S.mean()
    half_width = 1.96 * np.sqrt(p_est * (1 - p_est) / N)
    return p_est, half_width


def main():
    folder = Path("monte_carlo")
    gyro, gamma0, mount, truth = load_campaign(folder)
    N = len(truth)
    print("simulations :", N, " mesures par simulation :", gyro.shape[0])

    eps = np.zeros(N)
    err_I = np.zeros(N)
    err_axes = np.zeros(N)
    for k in range(N):
        # les 4 colonnes de la simulation k
        t, omega = gyro[:, 4*k], gyro[:, 4*k + 1:4*k + 4]
        m, a, *Ib = truth[k]
        eps[k], err_I[k], err_axes[k] = analyse_simulation(t, omega, gamma0[k], mount[k],
                                                          m, a, np.array(Ib))
        if (k + 1) % 500 == 0:
            print(f"{k + 1} / {N} simulations traitees")

    """
        Resultats
    """
    print("\nerreur de Frobenius eps :")
    print(f"  mediane      : {np.median(eps) * 100:.3f} %")
    print(f"  quantile 95% : {np.quantile(eps, 0.95) * 100:.3f} %")
    print(f"  max          : {np.max(eps) * 100:.3f} %")
    print("erreur relative max sur les moments, mediane :", f"{np.median(err_I) * 100:.3f} %")
    print("erreur max sur les axes, mediane :", f"{np.median(err_axes):.3f} deg")
    print("estimations echouees (Phi^T Phi singuliere) :", int(np.sum(np.isinf(eps))))

    print("\ntaux de reussite p = P(eps < tol), IC 95 % :")
    for tol in TOLS:
        p_est, hw = success_rate(eps, tol)
        print(f"  tol = {tol * 100:.0f} % : p = {p_est * 100:.2f} % +- {hw * 100:.2f} %")

    # une ligne par simulation, pour les figures et la sensibilite
    with (folder / "estimation_results.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sim", "eps_F", "err_I_max", "err_axes_max_deg"])
        writer.writerows([[k, eps[k], err_I[k], err_axes[k]] for k in range(N)])

    # histogramme de eps (echelle log : les erreurs vont sur plusieurs ordres de grandeur)
    finite = eps[np.isfinite(eps)]
    fig, ax = plt.subplots(figsize = (6, 4))
    ax.hist(finite * 100, bins = np.logspace(np.log10(finite.min() * 100),
                                             np.log10(finite.max() * 100), 50))
    ax.set_xscale("log")
    for tol in TOLS:
        ax.axvline(tol * 100, color = "tab:red", ls = "--", lw = 1)
    ax.set_xlabel(r"$\varepsilon$ (%)")
    ax.set_ylabel("nombre de simulations")
    ax.set_title(f"Erreur d'estimation de J sur {N} simulations")
    fig.tight_layout()
    fig.savefig(folder / "hist_eps.png", dpi = 150)
    print("\nresultats sauvegardes dans :", folder)


if __name__ == "__main__":
    main()
