"""
    Le but de ce truc c'est de simuler la sphere avec
    un capteur a l'interieur, et on extrait les donnees dans un csv
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from sensors.gyrometer import Gyrometer

def main():
    #sigma = 0.005, biais_std = 0.001, freq = 100
    # ecart type du bruit blanc du capteur
    sigma = 0.005
    # biais du capteur, pour l'instant constant, genere par N(0, biais_var I_3)
    biais_std = 0.001
    # frequence du capteur : MPU6050 --> 100Hz
    freq = 100 #Hz
    gyro = Gyrometer(sigma = sigma,
                     biais_std = biais_std,
                     freq = freq)
    # Initialiser l'engine physics
    physics = "quaternion" # ou "euler" si besoin de valider un truc
    
    # Initialiser la sphere et ses params
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
    # Params de la simulation 
    #---------------------------------------------------------------------#
    T = 5.0 #s temps de simulation
    h= 1e-3 #taille du pas
    N = (int)(T/h) # nombre de pas
    
    # tolerance du schema : arreter la simulation lorsque le drift des quantites conservees dépasse ce seuil d'erreur 
    eps_schema = 1e-6
    
    # donnees synthetiques du capteur
    out = Path("synthetic_data")
    out.mkdir(exist_ok = True)
    
    #------------------------------------------------------------------------------------------------------------------#
    # Lancement de la simulation
    #------------------------------------------------------------------------------------------------------------------#
    result = sphere.simulate(r0 = r0, 
                             alpha0 = alpha0,
                             Omega0 = Omega0,
                             h = h,
                             N = N,
                             eps_schema = eps_schema)
    print(sphere)
    print("simulation valide :", result.valid, result.stop_reason)

    #------------------------------------------------------------------------------------------------------------------#
    # Mesures du gyrometre, dans le repere du capteur
    #------------------------------------------------------------------------------------------------------------------#
    t_gyro, omega_gyro = gyro.mesure(result)

    csv_path = out / "gyro.csv"
    with csv_path.open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "omega_x", "omega_y", "omega_z"])   # s, rad/s
        for t_k, omega_k in zip(t_gyro, omega_gyro):
            writer.writerow([t_k, *omega_k])
    print(f"Donnees sauvegardees dans : {csv_path.resolve()}")

    # gamma0 dans le repere du gyro : ce que mesure l'accelerometre au repos (divise par g)
    gamma0 = gyro.mount.T @ result.R[0].T @ np.array([0.0, 0.0, 1.0])
    with (out / "gamma0.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma_x", "gamma_y", "gamma_z"])
        writer.writerow(gamma0)

    # montage du gyro : verite terrain, sert seulement a valider l'estimation
    with (out / "mount.csv").open("w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(gyro.mount)
    print("montage du gyro (Q, gyro -> corps) :\n", np.round(gyro.mount, 4))



if __name__ == "__main__":
    main()
