"""
    Cas b : boule lancee pres du blocage de cardan (tangage theta proche de 90 deg).

    A lancer depuis le dossier Python/ :  python -m scripts.run_case_b
    A theta = 90 deg exactement, E_s(alpha_0) n'est pas inversible : la
    simulation en angles d'Euler ne peut meme pas calculer alpha_dot_0.
    On lance donc la boule a theta_0 = 90 - DELTA deg, et on superpose les
    traces obtenues en angles d'Euler et en quaternions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import plotting
from chaplygin.controls import kinetic_energy, angular_momentum_P, relative_drift
from chaplygin.euler.simulation import simulate as simulate_euler

DELTA = 0.4     # ecart a 90 deg (deg)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("enregistre :", path)


def main():
    sphere = ChaplyginSphere(m=1.0, a=0.1, Ib=[1.5e-3, 8.5e-3, 9.5e-3],
                             physics="quaternion")
    r0 = np.zeros(2)
    alpha0 = np.array([0.0, np.radians(90 - DELTA), 0.0])
    Omega0 = np.array([5.0, 1.0, 2.0])     # rad/s, repere sol
    h, N = 1e-3, 5000
    out = Path("results")
    out.mkdir(exist_ok=True)

    result = sphere.simulate(r0, alpha0, Omega0, h, N)
    print(sphere)
    print("simulation valide :", result.valid, result.stop_reason)
    print("min |cos(theta)| :", np.abs(np.cos(result.alpha[:, 1])).min())

    # image de la boule le long de la trajectoire
    save(plotting.plot_trajectory_3d(sphere, result).figure,
         out / "case_b_trajectory_3d.png")

    # angles d'Euler : seulement pour la comparaison, sans les controles
    # (sinon la simulation s'arrete des que |cos(theta)| < eps)
    with np.errstate(all="ignore"):
        result_euler = simulate_euler(sphere, r0, alpha0, Omega0, h, N, check=False)
    gap = np.linalg.norm(result_euler.r - result.r, axis=1)
    print("ecart final |r_e - r_q| :", gap[-1])

    # trajectoire du centre et angles d'Euler, les deux solveurs superposes
    # (meme couleur pour une grandeur, trait plein = quaternions, tirets = Euler)
    results = {"quaternions": (result, "-"), "angles d'Euler": (result_euler, "--")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for k, (name, (res, ls)) in enumerate(results.items()):
        ax1.plot(res.r[:, 0], res.r[:, 1], ls, color=f"C{k}", label=name)
        # en degres, phi et psi ramenes dans [-180, 180]
        angles = np.degrees(res.alpha)
        angles[:, [0, 2]] = (angles[:, [0, 2]] + 180) % 360 - 180
        for j, angle in enumerate((r"\varphi", r"\theta", r"\psi")):
            ax2.plot(res.t, angles[:, j], ls, color=f"C{j}",
                     label=f"${angle}$ ({name})")
    ax1.plot(*r0, "o", color="tab:green", label="départ")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set(title="Trajectoire du centre", xlabel="x (m)", ylabel="y (m)")
    for theta_lock in (-90, 90):
        ax2.axhline(theta_lock, color="gray", lw=1, ls=":")
    ax2.set(title=r"Angles d'Euler (Gimbal lock en $\theta = \pm 90°$)",
            xlabel="t (s)", ylabel="deg", yticks=np.arange(-180, 181, 45))
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, ncol=1 if ax is ax1 else 2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out / "case_b_euler_vs_quat.png")

    # reactions et derives
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for name, (res, ls) in results.items():
        t, fr = res.t, res.friction
        for k, (force, values) in enumerate([(r"N_x", fr[:, 0]),
                                             (r"N_y", fr[:, 1]),
                                             (r"\|(N_x, N_y)\|",
                                              np.linalg.norm(fr, axis=1))]):
            axs[0].plot(t, values, ls, color=f"C{k}", label=f"${force}$ ({name})")
        dT = relative_drift(kinetic_energy(sphere, res))
        dL = relative_drift(angular_momentum_P(sphere, res))
        print(f"{name:15s} : derive max de T = {dT.max():.1e}, de L_P = {dL.max():.1e}")
        axs[1].semilogy(t[1:], dT[1:], ls, color="C0", label=f"$T$ ({name})")
        axs[1].semilogy(t[1:], dL[1:], ls, color="C1", label=f"$L_P$ ({name})")
    bound = sphere.mu_s * sphere.m * sphere.g
    axs[0].set(title=f"Réactions tangentielles (borne $\\mu_s m g$ = {bound:.1f} N)",
               ylabel="N")
    axs[1].set(title="Dérive relative des invariants", xlabel="t (s)")
    for ax in axs:
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, out / "case_b_controls.png")
    plt.show()


if __name__ == "__main__":
    main()
