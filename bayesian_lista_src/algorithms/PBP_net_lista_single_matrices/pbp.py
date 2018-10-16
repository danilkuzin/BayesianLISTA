import sys

import math

import numpy as np

import theano

import theano.tensor as T

from algorithms.PBP_net_lista_single_matrices import prior, network


class PBP_lista:

    def __init__(self, L, D, K, X_design_matrix, mean_y_train, std_y_train, initial_lambda):

        self.D = D
        self.K = K

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(L, D, K, var_targets)

        # We create the network

        params = self.prior.get_initial_params()

        thr_lambda = initial_lambda

        params['W_M'] = X_design_matrix.T / (1.01 * np.linalg.norm(X_design_matrix, 2) ** 2)
        params['S_M'] = np.identity(D) - np.matmul(params['W_M'], X_design_matrix)

        self.network = network.Network(params['W_M'], params['W_V'], params['S_M'], params['S_V'], thr_lambda,
                                       params['a'], params['b'], D, K, L)

        # We create the input and output variables in theano

        self.y = T.vector('y')
        self.beta = T.vector('beta')

        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2 = \
            self.network.logZ_Z1_Z2(self.beta, self.y)

        # We create a theano function for updating the posterior

        self.adf_update = theano.function([self.beta, self.y], self.logZ,
                                          updates=self.network.generate_updates(self.logZ, self.logZ1,
                                                                                self.logZ2))

        self.check_updates = theano.function([self.beta, self.y], T.grad(self.logZ, self.network.params_S_M))

        # We greate a theano function for the network predictive distribution

        self.predict_probabilistic = theano.function([self.y],
                                                     self.network.output_probabilistic(self.y))

        self.predict_deterministic = theano.function([self.y],
             self.network.output_deterministic(self.y))


        self.logs_output = theano.function([self.beta, self.y], [self.logZ, self.logZ1, self.logZ2])

    def do_pbp(self, Beta_train, y_train, n_iterations):

        if n_iterations > 0:

            # We first do a single pass

            self.do_first_pass(Beta_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            for i in range(int(n_iterations) - 1):
                # We do one more pass

                params = self.prior.get_params()
                self.do_first_pass(Beta_train, y_train)

                # We refine the prior

                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)


    def get_deterministic_output(self, Y_test):

        output = np.zeros((Y_test.shape[0], self.D))
        for i in range(Y_test.shape[0]):
            output[i] = self.predict_deterministic(Y_test[i, :])
            # output[i] = output[i] * self.std_y_train + self.mean_y_train

        return output

    def get_predictive_omega_mean_and_variance(self, Y_test):

        mean = np.zeros((Y_test.shape[0], self.D))
        variance = np.zeros((Y_test.shape[0], self.D))
        omega = np.zeros((Y_test.shape[0], self.D))

        for i in range(Y_test.shape[0]):
            w, m, v = self.predict_probabilistic(Y_test[i, :])
            # m = m * self.std_y_train + self.mean_y_train
            # v = v * self.std_y_train ** 2
            # print('prediction {0}\n. w:{1}\n, m:{2}\n, v:{3}\n. Input was:{4}\n'.format(i, w, m, v, Y_test[i, :]))
            mean[i] = m
            variance[i] = v
            omega[i] = w

        v_noise = self.network.b.get_value() / \
                  (self.network.a.get_value() - 1) * self.std_y_train ** 2

        return omega, mean, variance, v_noise

    def do_first_pass(self, Beta, Y):

        permutation = np.random.choice(range(Beta.shape[0]), Beta.shape[0],
                                       replace=False)

        counter = 0
        for i in permutation:
            w, m, v = self.predict_probabilistic(Y[i, :])
            logs = self.logs_output(Beta[i, :], Y[i, :])
            old_params = self.network.get_params()
            # ttt = self.check_updates(Beta[i, :], Y[i, :])
            # ttt1 = old_params['S_V']
            # ttt2 = old_params['S_M']
            # print('old mean:{}, var:{}, grad:{}'.format(ttt2[0, 0], ttt1[0, 0], ttt[0, 0]))
            Z = self.adf_update(Beta[i, :], Y[i, :])
            new_params = self.network.get_params()

            # s_new = new_params['S_M']
            # print(s_new[0 ,0 ])
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            counter += 1

    def sample_ws(self):

        self.network.sample_ws()

    def sample_mean_ws(self):
        self.network.sample_mean_ws()
