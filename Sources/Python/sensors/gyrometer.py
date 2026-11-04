"""
    Code pour simuler un gyrometre.
    
    Je prendrais un MPU6050 de protocol I2C
    par mes experiences, ce truc est normalement a 100Hz donc 10ms une mesure.
    
    Sur un gyrometre, il y a aussi du bruit et du biais
"""
import numpy as np

class Gyrometer:
    def __init__(self, sigma = 0.005, biais_std = 0.001, freq = 100,  seed = 42):
        """
            Mesure : 
                z_k = omega_{body, k} + b_k + nu_k
                
            b_k : biais a t = k, on peut avoir un biais fixe ou biais de marche aleatoire
            nu_k : bruit blanc pour la vitesse angulaire suivant les axes principaux du corps

                        nu_k ~ N(0, sigma^2 I_3)
        """
        self.rng = np.random.default_rng(seed)
        # ecart type du bruit 
        self.sigma = sigma
        
        # biais initial 
        self.biais = self.rng.normal(0, self.biais_std, 3)
        
        # frequence du capteur (Hz)
        self.freq = freq
        
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
        tick = round((1.0 / self.freq) / h)

        R, Omega = result.R[::tick], result.Omega[::tick]
        omega_body = np.einsum("nji,nj->ni", R, Omega)
        
        noise = self.rng.normal(0, self.sigma, omega_body.shape)
        # retour lorsque tickrate est atteint
        return result.t[::tick], omega_body + self.bias + noise

        

        