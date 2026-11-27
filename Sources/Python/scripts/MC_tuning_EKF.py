"""
    Ce code permet de lancer une campagne MC, afin de estimer la performance du filtre grace a la NIS
    On cherche a tuner le filtre jusqua ce que la NIS ait le comportement attendu

    On simule UNE trajectoire, puis pour chaque reglage (Q, R, P0) on fait N
    tirages capteurs independants (bruits, biais initiaux, montage de l'IMU).
    Critere : NIS moyenne dans la bande a 95 % du chi2(3N)/N, sans divergence.

    Reglages balayes (facteurs sur les valeurs "vraies" des capteurs) :
        k_q  : sigma_bg et sigma_ba declares au filtre   -> Q sur les biais
        k_r  : sigma_a declare au filtre                 -> R
        k_p0 : ecarts types initiaux des biais dans P0   (base = biais_std)

    Sorties : mc_tuning/resultats.csv, mc_tuning/anis.png, mc_tuning/anis_hist.png

    python -m scripts.MC_tuning_EKF
"""
import csv
import itertools
import math
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from sensors.gyrometer import Gyrometer
from sensors.accelerometer import Accelerometer
from filter.EKF import EKF, initial_state


#-------------------------------------------------------------------------#
# Capteurs (memes valeurs que real_simulate_sphere_collectdata.py)
#-------------------------------------------------------------------------#
freq = 100 #Hz
sigma_g, biais_std_g, sigma_bg = 0.05, 0.05, 5e-3
sigma_a, biais_std_a, sigma_ba = 0.5, 0.3, 7e-3

#-------------------------------------------------------------------------#
# Monte-Carlo et grille de reglage
#-------------------------------------------------------------------------#
N = 50                       # nombre de tirages par reglage
T = 50.0                     # s
K_Q = [1, 3, 10]
K_R = [1, 1.5, 2]
K_P0 = [1, 3]
sigma_gam0 = 0.1
seuil_div = 10.0             # deg : divergence si l'erreur sur gamma depasse ce seuil
t_trans = 10.0               # s : transitoire exclu de l'histogramme de la NIS


def simulate():
    m, a = 1.0, 0.1
    Ib = np.array([1.5e-3, 8.5e-3, 9.5e-3])
    sphere = ChaplyginSphere(m = m, a = a, Ib = Ib, physics = "quaternion")
    alpha0 = np.deg2rad([10.0, 20.0, 10.0])
    Omega0 = np.array([0.8, 0.10, 5.30])
    h = 1e-3
    result = sphere.simulate(r0 = np.zeros(2), alpha0 = alpha0, Omega0 = Omega0,
                             h = h, N = int(T / h), eps_schema = 1e-6)
    print("simulation valide :", result.valid, result.stop_reason)
    return sphere, result


def run_one(args):
    """
        Un tirage capteurs + un EKF. Renvoie la NIS (K,) et l'erreur d'angle (K,)
    """
    i, k_q, k_r, k_p0 = args
    gyro = Gyrometer(sigma_g = sigma_g, sigma_bg = sigma_bg, biais_std = biais_std_g,
                     freq = freq, mode = "realist", seed = 1000 + i)
    acc = Accelerometer(sigma_a = sigma_a, sigma_ba = sigma_ba, biais_std = biais_std_a,
                        freq = freq, mode = "realist", seed = 2000 + i, mount = gyro.mount)
    _, z_g = gyro.mesure(RESULT)
    _, z_a = acc.mesure(RESULT, SPHERE)

    x0, P0 = initial_state(z_a[0], sigma_gam0 = sigma_gam0,
                           sigma_bg0 = k_p0 * biais_std_g, sigma_ba0 = k_p0 * biais_std_a)
    ekf = EKF(a = SPHERE.a, g = SPHERE.g, dt = 1.0 / freq,
              sigma_g = sigma_g, sigma_bg = k_q * sigma_bg,
              sigma_a = k_r * sigma_a, sigma_ba = k_q * sigma_ba,
              x0 = x0, P0 = P0)
    xs, _, nis = ekf.run(z_g, z_a)

    cos = np.clip(np.sum(xs[:, :3] * acc.gamma_traj, axis = 1), -1, 1)
    return nis[1:], np.degrees(np.arccos(cos))[1:]


def init_worker(sphere, result):
    global SPHERE, RESULT
    SPHERE, RESULT = sphere, result


def main():
    out = Path("mc_tuning")
    out.mkdir(exist_ok = True)
    sphere, result = simulate()
    t = result.t[::round(1 / (freq * (result.t[1] - result.t[0])))][1:]

    # bande a 95 % de la NIS moyenne : N * ANIS ~ chi2(3N), approx. de Wilson-Hilferty
    def chi2_ppf(p, nu):
        z = {0.025: -1.959964, 0.975: 1.959964}[p]
        return nu * (1 - 2 / (9 * nu) + z * np.sqrt(2 / (9 * nu)))**3
    lo, hi = chi2_ppf(0.025, 3 * N) / N, chi2_ppf(0.975, 3 * N) / N
    print(f"bande a 95 % de la NIS moyenne (N = {N}) : [{lo:.3f}, {hi:.3f}]")

    reglages = list(itertools.product(K_Q, K_R, K_P0))
    rows, anis_all = [], {}
    with Pool(initializer = init_worker, initargs = (sphere, result)) as pool:
        for k_q, k_r, k_p0 in reglages:
            res = pool.map(run_one, [(i, k_q, k_r, k_p0) for i in range(N)])
            nis = np.array([r[0] for r in res])          # (N, K)
            ang = np.array([r[1] for r in res])          # (N, K)

            # divergence : erreur sur gamma > seuil sur la seconde moitie
            div = np.max(ang[:, ang.shape[1] // 2:], axis = 1) > seuil_div
            anis = nis[~div].mean(axis = 0) if (~div).any() else nis.mean(axis = 0)
            dedans = np.mean((anis > lo) & (anis < hi))
            rmse = np.sqrt(np.mean(ang[~div]**2)) if (~div).any() else np.nan

            anis_all[(k_q, k_r, k_p0)] = anis
            rows.append([k_q, k_r, k_p0, anis.mean(), 100 * dedans, div.sum(), rmse])
            print(f"k_q = {k_q:4}, k_r = {k_r:4}, k_p0 = {k_p0}  |  ANIS {anis.mean():6.2f}, "
                  f"{100 * dedans:5.1f} % dans la bande, {div.sum():2d}/{N} divergent, "
                  f"RMSE gamma {rmse:.2f} deg", flush = True)

    with (out / "resultats.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k_q", "k_r", "k_p0", "anis_moyenne", "pct_dans_bande",
                         "n_divergent", "rmse_gamma_deg"])
        writer.writerows(rows)

    #---------------------------------------------------------------------#
    # Figure : ANIS au cours du temps, une sous-figure par (k_q, k_r)
    #---------------------------------------------------------------------#
    fig, axes = plt.subplots(len(K_Q), len(K_R), figsize = (13, 9), sharex = True, sharey = True,
                             squeeze = False)
    for (iq, k_q), (ir, k_r) in itertools.product(enumerate(K_Q), enumerate(K_R)):
        ax = axes[iq, ir]
        for k_p0 in K_P0:
            ax.plot(t, anis_all[(k_q, k_r, k_p0)], lw = 0.6, label = f"$k_{{P_0}}$ = {k_p0}")
        ax.axhspan(lo, hi, color = "tab:red", alpha = 0.15)
        ax.axhline(3, color = "tab:red", lw = 0.8)
        ax.set_title(f"$k_Q$ = {k_q}, $k_R$ = {k_r}", fontsize = 10)
        ax.set_yscale("log")
        ax.grid(alpha = 0.3)
    axes[0, 0].legend(fontsize = 8)
    for ax in axes[-1]:
        ax.set_xlabel("t (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ANIS")
    fig.suptitle(f"NIS moyenne sur N = {N} tirages, bande a 95 % en rouge")
    fig.tight_layout()
    fig.savefig(out / "anis.png", dpi = 150)

    #---------------------------------------------------------------------#
    # Figure : frequence de l'ANIS (apres le transitoire) et densite theorique
    #   N ANIS ~ chi2(3N)  =>  densite de l'ANIS : f(x) = N f_chi2(3N)(N x)
    #---------------------------------------------------------------------#
    def chi2_pdf(x, nu):
        return np.exp((nu / 2 - 1) * np.log(x) - x / 2 - nu / 2 * np.log(2) - math.lgamma(nu / 2))

    x = np.linspace(0.5 * lo, 2 * hi, 400)
    m = t > t_trans
    fig, axes = plt.subplots(len(K_Q), len(K_R), figsize = (13, 9), sharex = True,
                             squeeze = False)
    for (iq, k_q), (ir, k_r) in itertools.product(enumerate(K_Q), enumerate(K_R)):
        ax = axes[iq, ir]
        for k_p0 in K_P0:
            ax.hist(anis_all[(k_q, k_r, k_p0)][m], bins = 60, range = (x[0], x[-1]), density = True,
                    histtype = "step", lw = 1.2, label = f"$k_{{P_0}}$ = {k_p0}")
        ax.plot(x, N * chi2_pdf(N * x, 3 * N), color = "tab:red", lw = 1.5, label = rf"$\chi^2({3 * N})/{N}$")
        ax.axvspan(lo, hi, color = "tab:red", alpha = 0.1)
        ax.set_title(f"$k_Q$ = {k_q}, $k_R$ = {k_r}", fontsize = 10)
        ax.grid(alpha = 0.3)
    axes[0, 0].legend(fontsize = 8)
    for ax in axes[-1]:
        ax.set_xlabel("ANIS")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequence")
    fig.suptitle(f"Frequence de la NIS moyenne (N = {N}, t > {t_trans:.0f} s) et densite theorique")
    fig.tight_layout()
    fig.savefig(out / "anis_hist.png", dpi = 150)
    print(f"Resultats sauvegardes dans : {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
