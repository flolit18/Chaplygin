"""
    Validation de l'ordre du schema RK4 (simulation en quaternions)

    On simule la meme boule avec les pas h, h/2, h/4, ... jusqu'au meme
    instant T, puis on compare les etats finaux y = (r, R, Omega) :
        e_h = y_{h/2}(T) - y_h(T)                 (ecart mesurable)
        rho = |e_h| / |e_{h/2}|  ~  2^4 = 16      (ordre 4)
        |y_h(T) - y(T)|  ~  16/15 |e_h|           (estimation de l'erreur)

    A lancer depuis le dossier Python de la source :

    python -m scripts.validation_rk4_order
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

from chaplygin import ChaplyginSphere


def save(fig, path):
    fig.savefig(path, dpi = 150, bbox_inches = "tight")
    print("enregistre : ", path)


def final_state(result):
    # etat final y(T) = (r, R, Omega) mis bout a bout dans un vecteur
    # on a droit de prendre que l'etat final car il comprend
    # deja les erreur sur le chemin
    return np.concatenate([result.r[-1], result.R[-1].ravel(), result.Omega[-1]])


def main():
    # Initialiser l'engine physics : validation en quaternions
    physics = "quaternion"

    # Initialiser la sphere et ses params (memes que simulate_sphere)
    # masse de la boule
    m = 1.0 #kg

    # rayon de la boule
    a = 0.1 #m

    # moment d'inertie suivant les axes principaux du corps de la boule
    Ib_1, Ib_2, Ib_3 = 1.5e-3, 8.5e-3, 9.5e-3
    Ib = np.array([Ib_1, Ib_2, Ib_3])

    # La sphere
    sphere = ChaplyginSphere(m = m,
                             a = a,
                             Ib = Ib,
                             physics = physics)

    # alpha0 : orientation initiale de la boule en angles d'Euler
    roll0 = np.deg2rad(10.0)
    pitch0 = np.deg2rad(20.0)
    yaw0 = np.deg2rad(10.0)
    alpha0 = np.array([roll0, pitch0, yaw0])

    # Omega0 : vitesse angulaire initiale de la sphere dans le repère monde
    Omega0 = np.array([0.8, 0.10, 5.30]) # rad/s

    # Position spatiale initiale
    r0 = np.array([0.0, 0.0])

    #---------------------------------------------------------------------#
    # Params de la validation
    #---------------------------------------------------------------------#
    # instant final commun a toutes les simulations
    T = 5.0 #s

    # pas h, h/2, h/4, ... : assez grands pour que l'erreur reste loin
    # de l'erreur machine (~1e-13), sinon rho ne veut plus rien dire
    h0 = 0.02
    n_raffinements = 6
    hs = h0 / 2**np.arange(n_raffinements + 1)

    out = Path("simulation_output")
    out.mkdir(exist_ok = True)

    #------------------------------------------------------------------------------------------------------------------#
    # Lancement des simulations
    #------------------------------------------------------------------------------------------------------------------#
    # check = False : on veut l'erreur du schema meme si elle depasse eps_schema
    ys = []
    for h in hs:
        N = int(round(T / h))
        result = sphere.simulate(r0 = r0,
                                 alpha0 = alpha0,
                                 Omega0 = Omega0,
                                 h = h,
                                 N = N,
                                 check = False)
        ys.append(final_state(result))
    print(sphere)

    # ecarts e_h = y_{h/2} - y_h, rapports rho et ordre p = log2(rho)
    e = np.array([np.linalg.norm(ys[k+1] - ys[k]) for k in range(len(hs) - 1)])
    rho = e[:-1] / e[1:]
    p = np.log2(rho)
    
    csv_path = out / "convergence_rk4.csv"
    print(f"{'h':>10} {'|e_h|':>12} {'16/15 |e_h|':>12} {'rho':>8} {'p':>6}")
    with csv_path.open( "w", newline="", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h", "|e_h|", "erreur_estimee_pas_h : 16/15|e_h|", "rho", "p"])
        for k in range(len(e)):
            ligne = [hs[k], e[k], (16 / 15) * e[k]]
            ligne_print = f"{hs[k]:10.5f} {e[k]:12.3e} {16/15 * e[k]:12.3e}"
            if k < len(rho):
                ligne.extend([rho[k], p[k]])
                ligne_print += f" {rho[k]:8.2f} {p[k]:6.2f}"
            else:
                ligne.extend(["", ""])
            print(ligne_print)
            writer.writerow(ligne)
    print(f"Donnees sauvegardees dans : {csv_path.resolve()}")
    #print(f"{'h':>10} {'|e_h|':>12} {'16/15 |e_h|':>12} {'rho':>8} {'p':>6}")
    #for k in range(len(e)):
    #    ligne = f"{hs[k]:10.5f} {e[k]:12.3e} {16/15 * e[k]:12.3e}"
    #    if k < len(rho):
    #        ligne += f" {rho[k]:8.2f} {p[k]:6.2f}"
    #    print(ligne)

    # Ecart |e_h| en fonction de h, avec une droite de pente 4 en reference
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs[:-1], e, "o-", label=r"$\|e_h\| = \|y_{h/2}(T) - y_h(T)\|$")
    ax.loglog(hs[:-1], e[0] * (hs[:-1] / hs[0])**4, "k--", label=r"pente 4")
    ax.set(xlabel="h (s)", ylabel="écart", title=f"Ordre du schéma RK4 (T = {T:g} s)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    save(fig, out / "ordre_rk4.png")


if __name__ == "__main__":
    main()
