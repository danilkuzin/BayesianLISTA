import numpy as np

from ..shared.soft_thresholding import soft_threshold


class Ista(object):
    def __init__(self, L, D, K, X, initial_lambda):
        self.L = L
        self.D = D
        self.K = K

        self.W = X.T / (1.01 * np.linalg.norm(X, 2) ** 2)
        self.S = np.identity(D) - np.matmul(self.W, X)
        self.thr_lambda = initial_lambda

    def __call__(self, y):
        b = np.dot(y, self.W.T)
        beta_estimator_history = [soft_threshold(b, self.thr_lambda)]
        for l in range(1, self.L):
            c = b + np.dot(beta_estimator_history[-1], self.S.T)
            beta_estimator_history.append(soft_threshold(c, self.thr_lambda))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

