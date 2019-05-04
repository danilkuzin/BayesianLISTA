import numpy as np

from ..shared.soft_thresholding import soft_threshold


class Fista(object):
    def __init__(self, L, D, K, X, initial_lambda):
        self.L = L
        self.D = D
        self.K = K

        self.W = X.T / (1.01 * np.linalg.norm(X, 2) ** 2)
        self.S = np.identity(D, dtype=np.float32) - np.matmul(self.W, X)
        self.thr_lambda = initial_lambda

    def __call__(self, y):
        b = np.dot(y, self.W.T)
        beta_estimator_history = [soft_threshold(b, self.thr_lambda)]
        z = [beta_estimator_history[-1]]
        t = [1.]
        for l in range(1, self.L):
            c = b + np.dot(z[-1], self.S.T)
            beta_estimator_history.append(soft_threshold(c, self.thr_lambda))
            t.append((1 + np.sqrt(1 + 4. * t[-1]**2))/2)
            z.append(beta_estimator_history[-1] + (t[-2]-1)/t[-1]*(beta_estimator_history[-1] - beta_estimator_history[-2]))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

