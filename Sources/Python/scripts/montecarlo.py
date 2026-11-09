"""
    Campagne Monte Carlo : generation des donnees 
    Meme principe que les simulations normales mais 
    avec les lois specifiques decrites dans le rapport sur les parametres 
    qui decrivent une simulation.

    Tire N scenarios X = (q0, Omega0, m, a, I_b, Q), garde seulement les
    scenarios admissibles, simule chacun puis enregistre dans monte_carlo/ :
    
        gyro.csv  : 4 colonnes par simulation, cote a cote :
                    t_k, omega_x_k, omega_y_k, omega_z_k  (repere du gyro)
        gamma0.csv : une ligne par simulation, verticale initiale vue par le gyro
        mount.csv : une ligne par simulation, montage Q (Q11 Q12 ... Q33)
        truth.csv : une ligne par simulation, m, a, I1, I2, I3, Omega0 (verite terrain)

    J'aurais aussi pu coder en JAX mais cela vaut pas trop la peine...
    
    python -m scripts.montecarlo
"""
import csv
from pathlib import Path
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin.rotations import euler_from_quat
from sensors.gyrometer import Gyrometer


def random_quaternion(rng):
    """
        4 gaussiennes normalisees -> quaternion uniforme sur S^3
        donc rotation uniforme sur SO(3)
    """
    q = rng.normal(0, 1, size = 4)
    return q / np.linalg.norm(q)


def draw_scenario(rng, Omega_min):
    """
        Tire un scenario jusqu'a ce qu'il soit admissible (hors non-glissement,
        qui se verifie pendant la simulation). Retourne aussi le nombre de rejets
        
        C'est un peu manuel comme methode de generation mais pour l'instnat, je n'ai pas
        encore d'idee pour une generation contrainte donc.
    """
    rejected = 0
    while True:
        m = rng.uniform(0.5, 2.0) # kg
        a = rng.uniform(0.05, 0.15) # m
        Omega0 = np.array([rng.uniform(-2.0, 2.0),
                           rng.uniform(-2.0, 2.0),
                           rng.uniform(-6.0, 6.0)]) # rad/s, repere sol
        lambdas = rng.uniform(0.1, 1.0, size = 3) # moments reduits I_i / (m a^2)
        q0 = random_quaternion(rng) # orientation initiale

        # inegalites d'inertie : lambda_i + lambda_j >= lambda_k
        triangle = lambdas.sum() >= 2 * lambdas.max()
        # masse dans la boule : somme des lambda <= 2
        in_ball = lambdas.sum() <= 2.0
        # mouvement assez excite
        excited = np.linalg.norm(Omega0) >= Omega_min

        if triangle and in_ball and excited:
            Ib = m * a**2 * lambdas
            return m, a, Omega0, Ib, q0, rejected
        rejected += 1


def write_rows(path, header, rows):
    with path.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():

    # Nombre de simulations gardees
    N = 3000
    rng = np.random.default_rng(seed = 42)

    # ecart type du bruit blanc du capteur
    sigma = 0.005
    # biais du capteur, pour l'instant constant, genere par N(0, biais_std^2 I_3)
    biais_std = 0.001
    # frequence du capteur : MPU6050 --> 100Hz
    freq = 100 #Hz

    # Initialiser l'engine physics
    physics = "quaternion" # ou "euler" si besoin de valider un truc

    # params de la simulation, comme dans la partie V
    T = 5.0 #s
    h = 1e-3 #taille du pas
    N_steps = int(round(T / h)) #nombre de pas
    eps_schema = 1e-6
    r0 = np.array([0.0, 0.0])
    Omega_min = 1.0 #rad/s

    gyro_cols, gamma0_rows, mount_rows, truth_rows = [], [], [], []
    rejected_draw, rejected_slip = 0, 0

    k = 0
    while k < N:
        """
            Generation des parametres
        """
        m, a, Omega0, Ib, q0, rej = draw_scenario(rng, Omega_min)
        rejected_draw += rej
        # le simulateur prend les angles d'Euler : q0 -> alpha0
        alpha0 = euler_from_quat(q0)

        sphere = ChaplyginSphere(m = m, a = a, Ib = Ib, physics = physics)
        result = sphere.simulate(r0 = r0,
                                 alpha0 = alpha0,
                                 Omega0 = Omega0,
                                 h = h,
                                 N = N_steps,
                                 eps_schema = eps_schema)
        # non-glissement et conservation de T, L_G.e_z verifies sur toute la simulation
        if not result.valid:
            rejected_slip += 1
            continue

        # un gyro par scenario : nouveau montage, nouveau biais, nouveau bruit
        gyro = Gyrometer(sigma = sigma,
                         biais_std = biais_std,
                         freq = freq,
                         seed = int(rng.integers(2**32)),
                         mount = None)
        t_gyro, omega_gyro = gyro.mesure(result)

        # gamma0 dans le repere du gyro : ce que mesure l'accelerometre au repos (divise par g)
        gamma0 = gyro.mount.T @ result.R[0].T @ np.array([0.0, 0.0, 1.0])

        gyro_cols.append(np.column_stack([t_gyro, omega_gyro]))
        gamma0_rows.append([k, *gamma0])
        mount_rows.append([k, *gyro.mount.ravel()])
        truth_rows.append([k, m, a, *Ib, *Omega0])

        k += 1
        if k % 100 == 0:
            print(f"{k} / {N} simulations")

    print("scenarios rejetes au tirage :", rejected_draw)
    print("scenarios rejetes pendant la simulation (glissement ou derive de T, L_G.e_z) :", rejected_slip)

    """
        Sauvegarde
    """
    out = Path("monte_carlo")
    out.mkdir(parents = True, exist_ok = True)

    # gyro : les simulations cote a cote, 4 colonnes chacune
    header = ",".join(f"t_{j},omega_x_{j},omega_y_{j},omega_z_{j}" for j in range(N))
    np.savetxt(out / "gyro.csv", np.hstack(gyro_cols), delimiter = ",",
               header = header, comments = "", fmt = "%.8e")

    write_rows(out / "gamma0.csv", ["sim", "gamma_x", "gamma_y", "gamma_z"], gamma0_rows)
    write_rows(out / "mount.csv", ["sim"] + [f"Q{i}{j}" for i in range(1, 4) for j in range(1, 4)],
               mount_rows)
    write_rows(out / "truth.csv", ["sim", "m", "a", "I1", "I2", "I3",
                                   "Omega0_x", "Omega0_y", "Omega0_z"], truth_rows)
    print("donnees sauvegardees dans :", out)


if __name__ == "__main__":
    main()
