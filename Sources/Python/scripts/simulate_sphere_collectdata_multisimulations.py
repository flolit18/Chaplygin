"""
    C'est pour un test 
    Simule la sphere pour une liste de cas (I_b, montage du gyro) et
    enregistre les donnees du gyro de chaque cas dans
        synthetic_data/<nom du cas>/
            gyro.csv   : t, omega_x, omega_y, omega_z (repere du gyro)
            gamma0.csv : verticale initiale vue par le gyro (accelerometre au repos)
            mount.csv  : montage Q du gyro (verite terrain, pour la validation)
            truth.csv : I_b vrai (verite terrain, pour la validation)

    A lancer depuis le dossier Python de la source :

    python -m scripts.simulate_sphere_collectdata_multisimulations
"""
import csv
from pathlib import Path
import numpy as np

from chaplygin import ChaplyginSphere
from sensors.gyrometer import Gyrometer

#------------------------------------------------------------------------------------------------------------------#
# Liste des cas : nom, moments principaux I_b (kg m^2), montage du gyro
# mount = np.eye(3) : gyro aligne sur les axes principaux
# mount = None      : montage quelconque, tire uniformement sur SO(3)
#------------------------------------------------------------------------------------------------------------------#
CASES = [
    ("reference_aligne",    [1.5e-3, 8.5e-3, 9.5e-3], np.eye(3)),
    ("reference_quelconque", [1.5e-3, 8.5e-3, 9.5e-3], None),
    ("boule_2_quelconque",  [3.0e-3, 5.0e-3, 7.0e-3], None),
]


def write_rows(path, header, rows):
    with path.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    # capteur : MPU6050 --> 100Hz
    sigma = 0.005       # ecart type du bruit blanc (rad/s)
    biais_std = 0.001   # biais constant, genere par N(0, biais_std^2 I_3)
    freq = 100          # Hz

    # boule et lancement, communs a tous les cas
    m = 1.0 #kg
    a = 0.1 #m
    alpha0 = np.deg2rad([10.0, 20.0, 10.0])
    Omega0 = np.array([0.8, 0.10, 5.30]) # rad/s
    r0 = np.array([0.0, 0.0])

    # params de la simulation
    T = 5.0 #s
    h = 1e-3 #taille du pas
    N = int(round(T / h)) #nombre de pas
    eps_schema = 1e-6

    out = Path("synthetic_data")

    for name, Ib, mount in CASES:
        sphere = ChaplyginSphere(m = m, a = a, Ib = np.array(Ib), physics = "quaternion")
        result = sphere.simulate(r0 = r0,
                                 alpha0 = alpha0,
                                 Omega0 = Omega0,
                                 h = h,
                                 N = N,
                                 eps_schema = eps_schema)
        print(name, ": simulation valide :", result.valid, result.stop_reason)

        # meme graine pour tous les cas : meme bruit et meme biais
        gyro = Gyrometer(sigma = sigma, biais_std = biais_std, freq = freq, mount = mount)
        t_gyro, omega_gyro = gyro.mesure(result)

        # gamma0 dans le repere du gyro : ce que mesure l'accelerometre au repos (divise par g)
        gamma0 = gyro.mount.T @ result.R[0].T @ np.array([0.0, 0.0, 1.0])

        folder = out / name
        folder.mkdir(parents = True, exist_ok = True)
        write_rows(folder / "gyro.csv", ["t", "omega_x", "omega_y", "omega_z"],
                   [[t_k, *omega_k] for t_k, omega_k in zip(t_gyro, omega_gyro)])
        write_rows(folder / "gamma0.csv", ["gamma_x", "gamma_y", "gamma_z"], [gamma0])
        write_rows(folder / "mount.csv", None, gyro.mount)
        write_rows(folder / "truth.csv", ["m", "a", "I1", "I2", "I3"], [[m, a, *Ib]])
        print("donnees sauvegardees dans :", folder)


if __name__ == "__main__":
    main()
