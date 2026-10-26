"""
    Cas isotrope : boule homogene, I_b = 2/5 m a^2 Id, meme lancer que le cas a.

    A lancer depuis le dossier Python/ :  python -m scripts.run_case_iso
    Solution exacte : I_P = I_G + m a^2 (Id - e_z e_z^T) est constante, donc
    L_P = I_P Omega conserve donne Omega = Omega_0, puis
    r(t) = r_0 + a (Omega_y, -Omega_x) t (ligne droite) et lambda = 0.
    On compare la simulation a cette solution.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import simulation, plotting
from chaplygin.physics import E_space


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("enregistre :", path)


def main():
    m, a = 1.0, 0.1
    sphere = ChaplyginSphere(m=m, a=a, Ib=[0.4 * m * a**2] * 3)
    r0 = np.zeros(2)
    alpha0 = np.array([0.8, 0.7, 0.0])
    Omega0 = np.array([-0.31, 0.10, 5.30])     # rad/s, repere sol
    h, N = 1e-3, 5000
    out = Path("results")
    out.mkdir(exist_ok=True)

    result = simulation.simulate(sphere, r0, alpha0, Omega0, h, N)
    print(sphere)
    print("simulation valide :", result.valid, result.stop_reason)

    # ecarts a la solution exacte
    Omega = np.array([E_space(al) @ ad for al, ad in zip(result.alpha, result.alpha_dot)])
    r_exact = r0 + a * np.outer(result.t, [Omega0[1], -Omega0[0]])
    print("max |Omega - Omega_0| :", np.abs(Omega - Omega0).max())
    print("max |r - r_exact|     :", np.linalg.norm(result.r - r_exact, axis=1).max())
    print("max |lambda|          :", np.linalg.norm(result.lam, axis=1).max())

    save(plotting.plot_trajectory(result).figure, out / "case_iso_trajectory.png")
    save(plotting.plot_trajectory_3d(sphere, result).figure,
         out / "case_iso_trajectory_3d.png")
    save(plotting.plot_orientation(sphere, result), out / "case_iso_orientation.png")
    save(plotting.plot_summary(sphere, result), out / "case_iso_summary.png")
    plt.show()


if __name__ == "__main__":
    main()
