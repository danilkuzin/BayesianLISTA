import numpy as np

class Backprop:
    def __init__(self):
        pass

    def dlogZ_dmz(self, b, w, v, m):
        return (1-w) * (b - m) * np.exp(-(b-m)**2/(2*(v + b_gamma/(a_gamma-1)))) / (np.sqrt(2 * np.pi)*(v + b_gamma/(a_gamma-1)))