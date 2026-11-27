"""
    Le but de ce truc c'est de simuler la sphere avec une IMU a l'interieur
    (gyrometre + accelerometre, biais qui derivent), et on extrait les donnees
    dans des csv (dossier realist_data)

    python -m scripts.real_simulate_sphere_collectdata
"""
import csv
from pathlib import Path
import numpy as np

from chaplygin import ChaplyginSphere
from sensors.gyrometer import Gyrometer
from sensors.accelerometer import Accelerometer


def write_csv(path, header, rows):
    with path.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    #---------------------------------------------------------------------#
    # Capteurs : MPU6050 --> 100 Hz, gyro et accelero sur la meme carte
    #---------------------------------------------------------------------#
    freq = 100 #Hz

    # gyrometre : bruit blanc (rad/s), biais initial (rad/s), derive (rad/s/sqrt(s))
    sigma_g = 0.1
    biais_std_g = 0.03
    sigma_bg = 5e-3
    gyro = Gyrometer(sigma_g = sigma_g,
                     sigma_bg = sigma_bg,
                     biais_std = biais_std_g,
                     freq = freq,
                     mode = "realist")

    # accelerometre : bruit blanc (m/s^2), biais initial (m/s^2), derive (m/s^2/sqrt(s))
    # meme montage que le gyro (meme IMU)
    sigma_a = 0.3
    biais_std_a = 0.1
    sigma_ba = 5e-3
    acc = Accelerometer(sigma_a = sigma_a,
                        biais_std = biais_std_a,
                        sigma_ba = sigma_ba,
                        freq = freq,
                        mode = "realist",
                        mount = gyro.mount)

    # Initialiser l'engine physics
    physics = "quaternion"

    # Initialiser la sphere et ses params
    m = 1.0 #kg
    a = 0.1 #m
    Ib_1, Ib_2, Ib_3 = 1.5e-3, 8.5e-3, 9.5e-3
    Ib = np.array([Ib_1, Ib_2, Ib_3])
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
    # Params de la simulation
    #---------------------------------------------------------------------#
    T = 100.0 #s, plus long pour voir les biais deriver
    h = 1e-3
    N = (int)(T/h)
    eps_schema = 1e-6

    out = Path("realist_data")
    out.mkdir(exist_ok = True)

    #---------------------------------------------------------------------#
    # Lancement de la simulation
    #---------------------------------------------------------------------#
    result = sphere.simulate(r0 = r0,
                             alpha0 = alpha0,
                             Omega0 = Omega0,
                             h = h,
                             N = N,
                             eps_schema = eps_schema)
    print(sphere)
    print("simulation valide :", result.valid, result.stop_reason)

    #---------------------------------------------------------------------#
    # Mesures, dans le repere capteur
    #---------------------------------------------------------------------#
    t, z_g = gyro.mesure(result)
    _, z_a = acc.mesure(result, sphere)

    write_csv(out / "gyro.csv", ["t", "omega_x", "omega_y", "omega_z"],      # s, rad/s
              [[t_k, *z_k] for t_k, z_k in zip(t, z_g)])
    write_csv(out / "accel.csv", ["t", "f_x", "f_y", "f_z"],                 # s, m/s^2
              [[t_k, *z_k] for t_k, z_k in zip(t, z_a)])

    # verite terrain dans le repere capteur : sert seulement a valider l'EKF
    truth = np.hstack([t[:, None], acc.gamma_traj, gyro.biais_traj, acc.biais_traj])
    write_csv(out / "truth.csv",
              ["t", "gamma_x", "gamma_y", "gamma_z",
               "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z"], truth)
    write_csv(out / "mount.csv", None, gyro.mount)

    # parametres dont l'EKF a besoin
    write_csv(out / "params.csv",
              ["a", "g", "freq", "sigma_g", "sigma_bg", "sigma_a", "sigma_ba"],
              [[a, sphere.g, freq, sigma_g, sigma_bg, sigma_a, sigma_ba]])

    print(f"Donnees sauvegardees dans : {out.resolve()}")
    print("montage de l'IMU (Q, capteur -> corps) :\n", np.round(gyro.mount, 4))


if __name__ == "__main__":
    main()
