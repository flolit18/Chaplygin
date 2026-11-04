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
        self.biais = self.rng.normal(0, self.biais, 3)
        
        # frequence du capteur 
        self.freq = freq
        
        # tick rate : sachant que on a plusieurs timeline dans une simu 
        #self.tick_rate = round((1.0 / self.freq) / dt)
        
    def mesure(self, omega_body, dt):
        """
            z_k = omega_{body, k} + b_k + nu_k
        
        ou` omega_{body, k} est la vraie vitesse angulaire de la simulation, 
        on la parse dedans pour simuler
        """
        self.elapsed += dt
        if self.elapsed < 1.0 / self.freq:
            return None
        
        self.elapsed -= 1.0 / self.freq
        
        #self.bias += (self.bias_walk * np.sqrt(1.0 / self.freq)* self.rng.normal(size=3))
        noise = self.rng.normal(0, self.sigma, 3)

        return omega_body + self.bias + noise
        