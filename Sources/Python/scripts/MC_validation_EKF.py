"""
    Campagne MC de validation de l'EKF avec le reglage retenu (MC_tuning_EKF.py)

    Meme principe que le reglage, mais sur des donnees jamais vues :
        - M trajectoires differentes (orientation et vitesse initiales tirees)
        - pour chacune, R tirages capteurs (bruits, biais initiaux, montage)
        - seeds differentes de celles du reglage
    soit N = M R simulations.

    Indicateurs :
        NIS moyenne  : N ANIS ~ chi2(3N)
        NEES moyenne : N ANEES ~ chi2(9N),  NEES = e^T P^-1 e, e = x_hat - x
        RMSE sur gamma (deg), b_g, b_a au cours du temps

    Sorties : mc_validation/resultats.csv, mc_validation/{coherence, rmse}.png

    python -m scripts.MC_validation_EKF
"""
import csv
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
# Reglage retenu
#-------------------------------------------------------------------------#
k_q, k_r, k_p0 = 10, 1, 1
sigma_gam0 = 0.1

#-------------------------------------------------------------------------#
# Monte-Carlo
#-------------------------------------------------------------------------#
M = 5                        # trajectoires
R = 10                       # tirages capteurs par trajectoire
T = 50.0                     # s
t_trans = 10.0               # s : transitoire exclu des statistiques
seuil_div = 10.0             # deg


def simulate(j):
    """
        Trajectoire j : orientation initiale dans +-30 deg, vitesse initiale
        autour de celle du rapport
    """
    rng = np.random.default_rng(500 + j)
    m, a = 1.0, 0.1
    Ib = np.array([1.5e-3, 8.5e-3, 9.5e-3])
    sphere = ChaplyginSphere(m = m, a = a, Ib = Ib, physics = "quaternion")
    alpha0 = np.deg2rad(rng.uniform(-30, 30, 3))
    Omega0 = np.array([0.8, 0.10, 5.30]) + rng.normal(0, 0.5, 3)
    h = 1e-3
    result = sphere.simulate(r0 = np.zeros(2), alpha0 = alpha0, Omega0 = Omega0,
                             h = h, N = int(T / h), eps_schema = 1e-6)
    return sphere, result


def run_one(args):
    """
        Trajectoire j, tirage i : renvoie NIS, NEES, erreur d'angle, erreurs de biais
    """
    j, i = args
    sphere, result = TRAJ[j]
    gyro = Gyrometer(sigma_g = sigma_g, sigma_bg = sigma_bg, biais_std = biais_std_g,
                     freq = freq, mode = "realist", seed = 7000 + 100 * j + i)
    acc = Accelerometer(sigma_a = sigma_a, sigma_ba = sigma_ba, biais_std = biais_std_a,
                        freq = freq, mode = "realist", seed = 8000 + 100 * j + i, mount = gyro.mount)
    _, z_g = gyro.mesure(result)
    _, z_a = acc.mesure(result, sphere)
    truth = np.hstack([acc.gamma_traj, gyro.biais_traj, acc.biais_traj])

    x0, P0 = initial_state(z_a[0], sigma_gam0 = sigma_gam0,
                           sigma_bg0 = k_p0 * biais_std_g, sigma_ba0 = k_p0 * biais_std_a)
    ekf = EKF(a = sphere.a, g = sphere.g, dt = 1.0 / freq,
              sigma_g = sigma_g, sigma_bg = k_q * sigma_bg,
              sigma_a = k_r * sigma_a, sigma_ba = k_q * sigma_ba,
              x0 = x0, P0 = P0)
    xs, Ps, nis = ekf.run(z_g, z_a)

    e = xs - truth
    nees = np.einsum("ni,ni->n", e, np.linalg.solve(Ps, e[:, :, None])[:, :, 0])
    cos = np.clip(np.sum(xs[:, :3] * truth[:, :3], axis = 1), -1, 1)
    ang = np.degrees(np.arccos(cos))
    return nis[1:], nees[1:], ang[1:], e[1:, 3:6], e[1:, 6:]


def init_worker(traj):
    global TRAJ
    TRAJ = traj


def chi2_ppf(p, nu):
    """ quantile du chi2 (approximation de Wilson-Hilferty, tres precise pour nu grand) """
    z = {0.025: -1.959964, 0.975: 1.959964}[p]
    return nu * (1 - 2 / (9 * nu) + z * np.sqrt(2 / (9 * nu)))**3


def chi2_pdf(x, nu):
    return np.exp((nu / 2 - 1) * np.log(x) - x / 2 - nu / 2 * np.log(2) - math.lgamma(nu / 2))


def main():
    out = Path("mc_validation")
    out.mkdir(exist_ok = True)
    N = M * R

    traj = []
    for j in range(M):
        sphere, result = simulate(j)
        print(f"trajectoire {j} : valide = {result.valid} {result.stop_reason}", flush = True)
        traj.append((sphere, result))
    t = traj[0][1].t[::round(1 / (freq * (traj[0][1].t[1] - traj[0][1].t[0])))][1:]

    with Pool(initializer = init_worker, initargs = (traj,)) as pool:
        res = pool.map(run_one, [(j, i) for j in range(M) for i in range(R)])
    nis = np.array([r[0] for r in res])
    nees = np.array([r[1] for r in res])
    ang = np.array([r[2] for r in res])
    ebg = np.array([r[3] for r in res])
    eba = np.array([r[4] for r in res])

    div = np.max(ang[:, ang.shape[1] // 2:], axis = 1) > seuil_div
    m = t > t_trans
    anis, anees = nis.mean(axis = 0), nees.mean(axis = 0)
    lo3, hi3 = chi2_ppf(0.025, 3 * N) / N, chi2_ppf(0.975, 3 * N) / N
    lo9, hi9 = chi2_ppf(0.025, 9 * N) / N, chi2_ppf(0.975, 9 * N) / N
    in3 = np.mean((anis[m] > lo3) & (anis[m] < hi3))
    in9 = np.mean((anees[m] > lo9) & (anees[m] < hi9))
    rmse_ang = np.sqrt(np.mean(ang**2, axis = 0))
    rmse_bg = np.sqrt(np.mean(np.sum(ebg**2, axis = 2), axis = 0))
    rmse_ba = np.sqrt(np.mean(np.sum(eba**2, axis = 2), axis = 0))

    print(f"N = {N} simulations ({M} trajectoires x {R} tirages), {div.sum()} divergent")
    print(f"ANIS  (t > {t_trans:.0f} s) : {anis[m].mean():.3f}, bande [{lo3:.3f}, {hi3:.3f}], {100 * in3:.1f} % dedans")
    print(f"ANEES (t > {t_trans:.0f} s) : {anees[m].mean():.3f}, bande [{lo9:.3f}, {hi9:.3f}], {100 * in9:.1f} % dedans")
    print(f"RMSE  (t > {t_trans:.0f} s) : gamma {np.sqrt(np.mean(ang[:, m]**2)):.2f} deg, "
          f"b_g {np.sqrt(np.mean(np.sum(ebg[:, m]**2, axis = 2))):.4f} rad/s, "
          f"b_a {np.sqrt(np.mean(np.sum(eba[:, m]**2, axis = 2))):.4f} m/s^2")

    with (out / "resultats.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "anis", "anees", "rmse_gamma_deg", "rmse_bg", "rmse_ba"])
        writer.writerows(np.column_stack([t, anis, anees, rmse_ang, rmse_bg, rmse_ba]))

    #---------------------------------------------------------------------#
    # Figure 1 : coherence, NIS et NEES moyennes (frequence et au cours du temps)
    #---------------------------------------------------------------------#
    fig, axes = plt.subplots(2, 2, figsize = (13, 8))
    for row, (val, lo, hi, dof, name) in enumerate([(anis, lo3, hi3, 3, "ANIS"),
                                                    (anees, lo9, hi9, 9, "ANEES")]):
        ax = axes[row, 0]
        x = np.linspace(1e-3, 2 * hi, 400)
        ax.hist(val[m], bins = 60, range = (x[0], x[-1]), density = True, color = "0.6", label = name)
        ax.plot(x, N * chi2_pdf(N * x, dof * N), color = "tab:red", label = rf"$\chi^2({dof * N})/{N}$")
        ax.axvspan(lo, hi, color = "tab:red", alpha = 0.1)
        ax.set_xlabel(f"{name} (t > {t_trans:.0f} s)")
        ax.set_ylabel("frequence")
        ax.legend()
        ax.grid(alpha = 0.3)

        ax = axes[row, 1]
        ax.plot(t, val, color = "tab:blue", lw = 0.5, label = name)
        ax.axhspan(lo, hi, color = "tab:red", alpha = 0.15, label = "bande a 95 %")
        ax.axhline(dof, color = "tab:red", lw = 0.8)
        ax.set_yscale("log")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(name)
        ax.legend(loc = "upper right")
        ax.grid(alpha = 0.3)
    fig.suptitle(f"Coherence du filtre, N = {N} simulations ({M} trajectoires x {R} tirages)")
    fig.tight_layout()
    fig.savefig(out / "coherence.png", dpi = 150)

    #---------------------------------------------------------------------#
    # Figure 2 : RMSE au cours du temps
    #---------------------------------------------------------------------#
    fig, axes = plt.subplots(3, 1, figsize = (8, 8), sharex = True)
    for ax, val, lab in zip(axes, [rmse_ang, rmse_bg, rmse_ba],
                            [r"$\gamma$ (deg)", r"$b_g$ (rad/s)", r"$b_a$ (m/s$^2$)"]):
        ax.plot(t, val, color = "tab:blue", lw = 1)
        ax.set_ylabel("RMSE " + lab)
        ax.grid(alpha = 0.3)
    axes[0].set_title(f"Erreur quadratique moyenne sur N = {N} simulations")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out / "rmse.png", dpi = 150)
    print(f"Resultats sauvegardes dans : {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
