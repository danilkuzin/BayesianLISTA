import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..listapbp import prior, network


class PBP_lista:

    def __init__(self, L, D, K, X_design_matrix, mean_y_train, std_y_train, initial_lambda):
        self.D = D
        self.K = K

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        self.prior = prior.Prior(L, D, K, var_targets)

        params = self.prior.get_initial_params()

        thr_lambda = initial_lambda

        params['W_M'] = X_design_matrix.T / (1.01 * np.linalg.norm(X_design_matrix, 2) ** 2)
        params['S_M'] = np.identity(D) - np.matmul(params['W_M'], X_design_matrix)

        self.network = network.Network(params['W_M'], params['W_V'], params['S_M'], params['S_V'], thr_lambda,
                                       params['a'], params['b'], D, K, L)

    def train(self, Beta_train, y_train, n_iterations):
        for i in range(int(n_iterations)):
            self.train_iter(Beta_train, y_train)
            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

    def get_deterministic_output(self, Y_test):
        output = self.network.output_deterministic(Y_test)

        return output

    def get_predictive_omega_mean_and_variance(self, Y_test):

        mean = np.zeros((Y_test.shape[0], self.D))
        variance = np.zeros((Y_test.shape[0], self.D))
        omega = np.zeros((Y_test.shape[0], self.D))

        for i in range(Y_test.shape[0]):
            w, m, v = self.network.output_probabilistic(Y_test[i, :])
            # m = m * self.std_y_train + self.mean_y_train
            # v = v * self.std_y_train ** 2
            # print('prediction {0}\n. w:{1}\n, m:{2}\n, v:{3}\n. Input was:{4}\n'.format(i, w, m, v, Y_test[i, :]))
            mean[i] = m
            variance[i] = v
            omega[i] = w

        v_noise = self.network.b.get_value() / \
                  (self.network.a.get_value() - 1) * self.std_y_train ** 2

        return omega, mean, variance, v_noise

    def train_iter(self, Beta, Y):

        # permutation = np.random.choice(range(Beta.shape[0]), Beta.shape[0],
        #                                replace=False)
        #
        # counter = 0
        # for i in permutation:
        #
        #     with tf.GradientTape() as t:
        #         logZ, logZ1, logZ2 = self.network.logZ_Z1_Z2(Beta[i, :], Y[i, :])
        #     self.network.generate_updates(logZ, logZ1, logZ2, t)
        #
        #     counter += 1

        with tf.GradientTape() as t:
            logZ, logZ1, logZ2 = self.network.logZ_Z1_Z2(Beta, Y)
        self.network.generate_updates(logZ, logZ1, logZ2, t)

    def sample_ws(self):

        self.network.sample_ws()

    def sample_mean_ws(self):
        self.network.sample_mean_ws()
