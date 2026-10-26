"""
    Cas a : une boule anisotrope lancee avec une rotation quelconque.

    A lancer depuis le dossier Python/ :  python -m scripts.run_case_a
    Enregistre dans results/ les trajectoires 2D et 3D, les courbes et le GIF.
    Option --no-gif pour sauter l'animation (la plus lente a enregistrer).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import simulation, plotting


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("enregistre :", path)


def main():
    sphere = ChaplyginSphere(m=1.0, a=0.1, Ib=[1.5e-3, 8.5e-3, 9.5e-3])
    r0 = np.zeros(2)
    alpha0 = np.array([0.8, 0.7, 0.0])
    Omega0 = np.array([-0.31, 0.10, 5.30])     # rad/s, repere sol
    h, N = 1e-3, 5000
    out = Path("results")
    out.mkdir(exist_ok=True)

    result = simulation.simulate(sphere, r0, alpha0, Omega0, h, N)
    print(sphere)
    print("simulation valide :", result.valid, result.stop_reason)

    # trajectoires 2D et 3D
    save(plotting.plot_trajectory(result).figure, out / "case_a_trajectory.png")
    save(plotting.plot_trajectory_3d(sphere, result).figure,
         out / "case_a_trajectory_3d.png")

    # orientation : boules ramenees a l origine
    save(plotting.plot_orientation(sphere, result), out / "case_a_orientation.png")

    # courbes
    save(plotting.plot_angles(result).figure, out / "case_a_angles.png")
    save(plotting.plot_reactions(sphere, result).figure, out / "case_a_reactions.png")
    save(plotting.plot_invariants(sphere, result).figure, out / "case_a_invariants.png")
    save(plotting.plot_summary(sphere, result), out / "case_a_summary.png")

    # animation
    fig_anim, anim = plotting.animate_3d(sphere, result, every=20)
    if "--no-gif" not in sys.argv:
        anim.save(out / "case_a.gif", writer="pillow", fps=25, dpi=80)
        print("enregistre :", out / "case_a.gif")
    plt.show()


if __name__ == "__main__":
    main()
