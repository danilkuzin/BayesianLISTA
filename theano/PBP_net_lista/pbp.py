import sys

import math

import numpy as np

import theano

import theano.tensor as T

import network

import prior


class PBP_lista:

    def __init__(self, L, D, K, mean_y_train, std_y_train):

        self.D = D
        self.K = K

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(L, D, K, var_targets)

        # We create the network

        params = self.prior.get_initial_params()

        thr_lambda = 0.1

        self.network = network.Network(params['W_M'], params['W_V'], params['S_M'], params['S_V'], thr_lambda,
                                       params['a'], params['b'], D, K)

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

        # We greate a theano function for the network predictive distribution

        self.predict_probabilistic = theano.function([self.y],
                                                     self.network.output_probabilistic(self.y))

        self.predict_deterministic = theano.function([self.y],
             self.network.output_deterministic(self.y))

    def do_pbp(self, Beta_train, y_train, n_iterations):

        if n_iterations > 0:

            # We first do a single pass

            self.do_first_pass(Beta_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            sys.stdout.write('{}\n'.format(0))
            sys.stdout.flush()

            for i in range(int(n_iterations) - 1):
                # We do one more pass

                params = self.prior.get_params()
                self.do_first_pass(Beta_train, y_train)

                # We refine the prior

                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

                sys.stdout.write('{}\n'.format(i + 1))
                sys.stdout.flush()

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

            old_params = self.network.get_params()
            Z = self.adf_update(Beta[i, :], Y[i, :])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            if counter % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

            counter += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def sample_ws(self):

        self.network.sample_ws()
