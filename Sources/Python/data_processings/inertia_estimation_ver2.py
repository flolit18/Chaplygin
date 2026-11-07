"""
    Estimation de l'inertie pour tous les cas de synthetic_data/
    (generes par scripts.simulate_sphere_collectdata_multisimulations), en un seul lancement.

    Pour chaque cas : integration de gamma, Phi, moindres carres, diagonalisation
    de J, puis comparaison avec la verite terrain (truth.csv, mount.csv).
    Le resume est ecrit dans synthetic_data/estimation_summary.csv

    A lancer depuis le dossier Python de la source :

    python -m data_processings.inertia_estimation_ver2
"""
import csv
from pathlib import Path

import numpy as np

from data_processings.inertia_estimation import (load_gyro_csv, integrate_gamma, big_phi,
                                                 estimate_theta, params_from_theta)


def analyse_case(folder):
    """
        Estime J dans le repere du gyro pour un cas, et la compare a la verite terrain
    """
    m, a, *Ib = np.loadtxt(folder / "truth.csv", delimiter = ",", skiprows = 1)
    Ib = np.array(Ib)
    Q = np.loadtxt(folder / "mount.csv", delimiter = ",")
    gamma0 = np.loadtxt(folder / "gamma0.csv", delimiter = ",", skiprows = 1)
    t, omega = load_gyro_csv(folder / "gyro.csv")

    # estimation
    gamma = integrate_gamma(t, omega, gamma0)
    Phi, y = big_phi(m, a, gamma, omega)
    J_est, c_est, T_est = params_from_theta(estimate_theta(Phi, y))
    I_est, axes_est = np.linalg.eigh(J_est)     # valeurs propres croissantes

    # verite terrain, rangee dans le meme ordre (moments croissants)
    order = np.argsort(Ib)
    Ib, axes_true = Ib[order], Q.T[:, order]     # axes principaux vus du gyro = colonnes de Q^T

    err_I = np.abs(I_est - Ib) / Ib
    # angle entre axe estime et axe vrai, au signe pres
    cos_axes = np.abs(np.sum(axes_est * axes_true, axis = 0))
    err_axes = np.degrees(np.arccos(np.clip(cos_axes, 0, 1)))
    return Ib, I_est, err_I, err_axes


def main():
    data = Path("synthetic_data")
    folders = sorted(f for f in data.iterdir() if (f / "gyro.csv").exists())

    rows = []
    for folder in folders:
        Ib, I_est, err_I, err_axes = analyse_case(folder)
        rows.append([folder.name, *Ib, *I_est, *err_I, *err_axes])

        print(f"\n{folder.name}")
        print("  I vrai    (1e-3) :", np.round(Ib * 1e3, 3))
        print("  I estime  (1e-3) :", np.round(I_est * 1e3, 3))
        print("  erreur relative  :", np.round(err_I * 100, 2), "%")
        print("  erreur axes      :", np.round(err_axes, 3), "deg")

    summary = data / "estimation_summary.csv"
    with summary.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cas", "I1", "I2", "I3", "I1_est", "I2_est", "I3_est",
                         "err_I1", "err_I2", "err_I3", "err_axe1_deg", "err_axe2_deg", "err_axe3_deg"])
        writer.writerows(rows)
    print("\nresume sauvegarde dans :", summary)


if __name__ == "__main__":
    main()
