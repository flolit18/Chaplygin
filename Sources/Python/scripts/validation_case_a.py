"""
    Validation 1 : on relance la boule du cas a (angles d'Euler) avec le
    solveur en quaternions, et on compare les deux simulations.

    A lancer depuis le dossier Python/ :  python -m scripts.validation_case_a
    Enregistre dans results/ les ecarts entre les deux solveurs et le resume
    de la simulation en quaternions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import plotting
from chaplygin.controls import kinetic_energy, relative_drift


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("enregistre :", path)


def main():
    # meme lancer que le cas a
    sphere = ChaplyginSphere(m=1.0, a=0.1, Ib=[1.5e-3, 8.5e-3, 9.5e-3])
    r0 = np.zeros(2)
    alpha0 = np.array([0.8, 0.7, 0.0])
    Omega0 = np.array([-0.31, 0.10, 5.30])     # rad/s, repere sol
    h, N = 1e-3, 5000
    out = Path("results")
    out.mkdir(exist_ok=True)

    results = {}
    for physics in ("euler", "quaternion"):
        sphere.physics = physics
        results[physics] = sphere.simulate(r0, alpha0, Omega0, h, N)
        print(physics, ": simulation valide :", results[physics].valid,
              results[physics].stop_reason)
    res_e, res_q = results["euler"], results["quaternion"]

    # ecarts entre les deux solveurs a chaque pas
    t = res_q.t
    err_r = np.linalg.norm(res_e.r - res_q.r, axis=1)
    err_R = np.linalg.norm(res_e.R - res_q.R, axis=(1, 2))
    err_F = np.linalg.norm(res_e.friction - res_q.friction, axis=1)
    print("max |r_e - r_q|     :", err_r.max())
    print("max |R_e - R_q|     :", err_R.max())
    print("max |F_e - F_q|     :", err_F.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogy(t, err_r, label=r"$\|r_e - r_q\|$ (m)")
    ax1.semilogy(t, err_R, label=r"$\|R_e - R_q\|_F$")
    ax1.semilogy(t, err_F, label=r"$\|F_e - F_q\|$ (N)")
    ax1.set(title="Ecart Euler / quaternions", xlabel="t (s)")
    for name, res in results.items():
        dT = relative_drift(kinetic_energy(sphere, res))
        print(f"derive max de T ({name}) :", dT.max())
        ax2.semilogy(t[1:], dT[1:], label=name)
    ax2.set(title="Derive relative de l'energie", xlabel="t (s)")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    save(fig, out / "validation_case_a_errors.png")
    save(plotting.plot_summary(sphere, res_q), out / "validation_case_a_summary.png")
    plt.show()


if __name__ == "__main__":
    main()
