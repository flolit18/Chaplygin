"""
    EKF sur les donnees realistes (gyro + accelero, biais qui derivent)

    Entrees  : realist_data/{gyro, accel, truth, params}.csv
               (generees par scripts/real_simulate_sphere_collectdata.py)
    Sorties  : realist_filtered_data/filtered.csv : t, gamma_hat, bg_hat, ba_hat
               realist_filtered_data/{gamma, bg, ba, nis}.png

    Figures : capteur en points gris, estimation EKF en bleu, verite en rouge

    python -m data_processings.realist_filtered_data
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filter.EKF import EKF, initial_state


def load(folder):
    """
        Retourne t, z_g, z_a, truth (N, 9) = (gamma, b_g, b_a) et params (dict)
    """
    gyro = np.loadtxt(folder / "gyro.csv", delimiter = ",", skiprows = 1)
    accel = np.loadtxt(folder / "accel.csv", delimiter = ",", skiprows = 1)
    truth = np.loadtxt(folder / "truth.csv", delimiter = ",", skiprows = 1)
    with (folder / "params.csv").open(encoding = "utf-8") as f:
        rows = list(csv.reader(f))
    params = {k: float(v) for k, v in zip(rows[0], rows[1])}
    return gyro[:, 0], gyro[:, 1:], accel[:, 1:], truth[:, 1:], params


def plot_compare(t, est, true, title, ylabel, path, meas = None):
    """
        3 sous-figures (x, y, z) : mesure en points, EKF en bleu, verite en rouge
    """
    fig, axes = plt.subplots(3, 1, figsize = (8, 7), sharex = True)
    for i, (ax, comp) in enumerate(zip(axes, "xyz")):
        if meas is not None:
            ax.scatter(t, meas[:, i], s = 2, color = "0.6", label = "capteur")
        ax.plot(t, est[:, i], color = "tab:blue", lw = 1.2, label = "EKF")
        ax.plot(t, true[:, i], color = "tab:red", lw = 1.2, label = "vrai")
        ax.set_ylabel(f"{ylabel}$_{comp}$")
        ax.grid(alpha = 0.3)
    axes[0].legend(loc = "upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    return fig


def plot_nis(t, nis, path):
    """
        NIS_k = y_k^T S_k^-1 y_k ~ chi2(3) si le filtre est coherent

        gauche : frequence (histogramme normalise) et densite du chi2(3)
        droite : NIS au cours du temps, moyenne et intervalle a 95 %
    """
    nis, t = nis[1:], t[1:]          # nis[0] = nan (pas de correction a t = 0)
    lo, hi = 0.2158, 9.3484          # quantiles 2,5 % et 97,5 % du chi2(3)
    inside = np.mean((nis > lo) & (nis < hi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

    # frequence
    x = np.linspace(1e-3, max(15, np.percentile(nis, 99)), 300)
    ax1.hist(nis, bins = 80, range = (0, x[-1]), density = True, color = "0.6", label = "NIS")
    ax1.plot(x, np.sqrt(x) * np.exp(-x / 2) / np.sqrt(2 * np.pi), color = "tab:red", label = r"$\chi^2(3)$")
    ax1.set_xlabel("NIS")
    ax1.set_ylabel("frequence")
    ax1.legend()
    ax1.grid(alpha = 0.3)

    # NIS au cours du temps
    ax2.plot(t, nis, color = "tab:blue", lw = 0.5, label = "NIS")
    ax2.axhline(3, color = "tab:red", label = "moyenne attendue (3)")
    ax2.axhline(nis.mean(), color = "k", ls = "--", label = f"moyenne ({nis.mean():.2f})")
    ax2.axhspan(lo, hi, color = "tab:red", alpha = 0.1, label = f"95 % ({100 * inside:.1f} % dedans)")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("NIS")
    ax2.legend(loc = "upper right")
    ax2.grid(alpha = 0.3)

    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    return fig


def main():
    folder = Path("realist_data")
    out = Path("realist_filtered_data")
    out.mkdir(exist_ok = True)

    t, z_g, z_a, truth, p = load(folder)

    #------------------------------------------------------------------#
    # EKF
    #------------------------------------------------------------------#
    x0, P0 = initial_state(z_a[0])
    ekf = EKF(a = p["a"], g = p["g"], dt = 1.0 / p["freq"],
              sigma_g = p["sigma_g"], sigma_bg = p["sigma_bg"],
              sigma_a = p["sigma_a"], sigma_ba = p["sigma_ba"],
              x0 = x0, P0 = P0)
    xs, Ps, nis = ekf.run(z_g, z_a)

    with (out / "filtered.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "gamma_x", "gamma_y", "gamma_z",
                         "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z"])
        writer.writerows(np.hstack([t[:, None], xs]))

    #------------------------------------------------------------------#
    # Figures
    #------------------------------------------------------------------#
    # verticale "mesuree" par l'accelero : z_a / |z_a|
    gamma_meas = z_a / np.linalg.norm(z_a, axis = 1, keepdims = True)
    plot_compare(t, xs[:, :3], truth[:, :3], "Verticale", r"$\gamma$",
                 out / "gamma.png", meas = gamma_meas)
    plot_compare(t, xs[:, 3:6], truth[:, 3:6], "Biais du gyrometre", r"$b_g$ (rad/s) ",
                 out / "bg.png")
    plot_compare(t, xs[:, 6:], truth[:, 6:], "Biais de l'accelerometre", r"$b_a$ (m/s$^2$) ",
                 out / "ba.png")
    plot_nis(t, nis, out / "nis.png")

    # erreur d'angle entre gamma_hat et gamma
    cos = np.clip(np.sum(xs[:, :3] * truth[:, :3], axis = 1), -1, 1)
    print(f"erreur d'angle sur gamma : moyenne {np.degrees(np.arccos(cos)).mean():.3f} deg, "
          f"finale {np.degrees(np.arccos(cos[-1])):.3f} deg")
    print(f"Resultats sauvegardes dans : {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
