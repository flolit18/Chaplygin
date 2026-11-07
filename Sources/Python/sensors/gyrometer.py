"""
    Code pour simuler un gyrometre.
    
    Je prendrais un MPU6050 de protocol I2C
    par mes experiences, ce truc est normalement a 100Hz donc 10ms une mesure.
    
    Sur un gyrometre, il y a aussi du bruit et du biais
"""
import numpy as np

from chaplygin.rotations import quat_to_matrix

class Gyrometer:
    def __init__(self, sigma = 0.005, biais_std = 0.001, freq = 100,  seed = 42, mount = None):
        """
            Mesure : 
                z_k = omega_{body, k} + b_k + nu_k
                
            b_k : biais a t = k, on peut avoir un biais fixe ou biais de marche aleatoire
            nu_k : bruit blanc pour la vitesse angulaire suivant les axes du gyro

                        nu_k ~ N(0, sigma^2 I_3)

            mount : rotation Q du repere du gyro vers le repere corps (axes principaux),
                    omega_gyro = Q^T omega_corps. Si None, montage quelconque :
                    Q tiree uniformement sur SO(3)
        """
        self.rng = np.random.default_rng(seed)
        # ecart type du bruit 
        self.sigma = sigma
        
        # biais initial 
        self.biais = self.rng.normal(0, biais_std, 3)
        
        # frequence du capteur (Hz)
        self.freq = freq

        # montage du gyro dans la boule : quaternion unitaire uniforme
        # (4 gaussiennes normalisees -> loi uniforme sur la sphere S^3)
        if mount is None:
            q = self.rng.normal(size = 4)
            mount = quat_to_matrix(q / np.linalg.norm(q))
        self.mount = np.asarray(mount, dtype = float)
        
        # tick rate : sachant que on a plusieurs timeline dans une simu 
        #self.tick_rate = round((1.0 / self.freq) / dt)
        
    def mesure(self, result):
        """
            z_k = omega_{body, k} + b_k + nu_k
        
        ou` omega_{body, k} est la vraie vitesse angulaire de la simulation, 
        on la parse dedans pour simuler
        
        aux instants du capteur : on prend un pas de la simulation tous les
        tick = (1 / freq) / h pas, ou h est le pas de la simulation
        """
        h = result.t[1] - result.t[0]
        # tick rate
        ratio = (1.0 / self.freq) / h
        tick = round(ratio)
        # h doit diviser la periode du capteur, sinon la frequence change
        if tick < 1 or abs(ratio - tick) > 1e-9:
            raise ValueError(f"h = {h} doit diviser la periode du capteur 1/freq = {1/self.freq}")

        R, Omega = result.R[::tick], result.Omega[::tick]
        omega_body = np.einsum("nji,nj->ni", R, Omega)
        # passage au repere du gyro : omega_gyro = Q^T omega_corps
        omega_gyro = omega_body @ self.mount
        
        noise = self.rng.normal(0, self.sigma, omega_body.shape)
        # retour lorsque tickrate est atteint
        return result.t[::tick], omega_gyro + self.biais + noise

        

        