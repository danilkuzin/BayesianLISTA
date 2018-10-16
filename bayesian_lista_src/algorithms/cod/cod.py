import numpy as np
from algorithms.PBP_net_lista.test_network_layer import soft_threshold


class Cod(object):
    """
    Implements Coordinate Descent as in Gregor, LeCun 2010
    """
    def __init__(self, L, D, K, X, initial_lambda, threshold):
        self.L = L
        self.D = D
        self.K = K

        self.X = X


        self.S = np.matmul(X.T, X)
        self.thr_lambda = initial_lambda

        self.threshold = threshold

    def predict_full(self, y):

        beta_estimator = np.zeros(self.D)
        beta_estimator_history = [beta_estimator]
        b = np.dot(self.X.T, y)

        old_beta_estimator = np.zeros_like(beta_estimator)
        np.copyto(old_beta_estimator, beta_estimator)
        diff = 1e10
        while diff > self.threshold:
            beta_hat = soft_threshold(b, self.thr_lambda)
            k = np.argmax(np.abs(beta_hat - beta_estimator))
            for j in range(self.D):
                # if j == k:
                #     continue
                b[j] = b[j] - self.S[j, k] * np.abs(beta_hat[k] - beta_estimator[k])
            beta_estimator[k] = beta_hat[k]
            diff = np.sum((old_beta_estimator - beta_estimator)**2)

            beta_estimator_history.append(beta_estimator)
            np.copyto(old_beta_estimator, beta_estimator)

        beta_estimator = soft_threshold(b, self.thr_lambda)
        return beta_estimator
