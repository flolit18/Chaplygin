"""
    Lancer une simulation de la boule sur un plan
    
    A lancer depuis le dossier Python de la source : 
    
    python -m scripts.simulate_sphere
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from chaplygin import ChaplyginSphere
from chaplygin import plotting
from chaplygin.controls import kinetic_energy, relative_drift

def save(fig, path):
    fig.savefig(path, dpi = 150, bbox_inches = "tight")
    print("enregistre : ", path)
    
def main():
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
    h, N = 1e-3, 5000 #taille du pas, nombre de pas
    
    # tolerance du schema : arreter la simulation lorsque le drift des quantites conservees dépasse ce seuil d'erreur 
    eps_schema = 1e-6
    
    out = Path("results")
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

    # Angles d'Euler (phi, theta, psi) calcules a partir de R
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotting.plot_angles(result, ax=ax)
    save(fig, out / "angles_euler.png")

    # Animation de la boule, une image tous les every pas, en temps reel
    every = 20
    fig, anim = plotting.animate_3d(sphere, result, every=every)
    anim.save(out / "boule.gif", writer="pillow", fps=round(1 / (every * h)))
    print("enregistre : ", out / "boule.gif")


if __name__ == "__main__":
    main()
