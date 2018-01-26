import numpy as np


class DataGenerator:

    def __init__(self, D, K, beta_scale=1, noise_scale=0.1):
        self.D = D  # beta
        self.K = K  # y
        self.beta_scale = beta_scale
        self.noise_scale = noise_scale

        self.X = np.random.randn(K, D)

    def new_sample(self, N):
        Y = np.zeros((N, self.K))
        Beta = np.random.normal(loc=0, scale=self.beta_scale, size=(N, self.D))
        Noise = np.random.normal(loc=0, scale=self.noise_scale, size=(N, self.K))
        for n in np.arange(N):
            Y[n] = np.dot(self.X, Beta[n]) + Noise[n]

        return Beta, Y, Noise
