import numpy as np

class DataGenerator:

    def __init__(self, D, K, sparsity=0.8, beta_scale=1, noise_scale=0.5):
        self.D = D  # beta
        self.K = K  # y
        self.beta_scale = beta_scale
        self.noise_scale = noise_scale
        self.sparsity = sparsity

        self.X = np.random.randn(K, D)
        #X_distr = tfd.Normal(loc=tf.zeros(K, D), scale=tf.ones(K, D))
        #self.X = X_distr.sample([1])

    def sample_slab(self, N, borders=(-0.1, 0.1)):
        slab = np.zeros((N, self.D))
        for i1 in range(N):
            for i2 in range(self.D):
                sampled = False
                while not sampled:
                    sample = np.random.normal(loc=0, scale=self.beta_scale, size=None)
                    if sample < borders[0] or sample > borders[1]:
                        slab[i1, i2] = sample
                        sampled = True
        return slab

    def sample_omega(self, N):
        omega = np.zeros((N, self.D))
        eps = 1e-3
        for n in range(N):
            sampled = False
            while not sampled:
                omega[n] = np.random.binomial(1, self.sparsity, size=(1, self.D))
                if np.sum(1-omega[n]) > eps and np.sum(omega[n]) > eps:
                    sampled = True
        return omega


    def new_sample(self, N):
        Y = np.zeros((N, self.K))
        omega = self.sample_omega(N)
        Beta = (1-omega) * self.sample_slab(N)
        Noise = np.random.normal(loc=0, scale=self.noise_scale, size=(N, self.K))
        for n in np.arange(N):
            Y[n] = np.dot(self.X, Beta[n]) + Noise[n]

        return Beta, Y, Noise
