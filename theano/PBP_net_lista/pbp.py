import sys

import math

import numpy as np

import theano

import theano.tensor as T
from tqdm import tqdm

from PBP_net_lista import prior, network


class PBP_lista:

    def __init__(self, L, D, K, X_design_matrix, mean_y_train, std_y_train):

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

        params['W_M'] = []
        params['S_M'] = []

        W_init = X_design_matrix.T / (1.01 * np.linalg.norm(X_design_matrix, 2) ** 2)
        S_init = np.identity(D) - np.matmul(W_init, X_design_matrix)

        for i in range(L):
            params['W_M'].append(W_init)
            params['S_M'].append(S_init)

        self.network = network.Network(params['W_M'], params['W_V'], params['S_M'], params['S_V'], thr_lambda,
                                       params['a'], params['b'], D, K)

        # We create the input and output variables in theano

        self.y = T.vector('y')
        self.beta = T.vector('beta')

        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2, self.test_student, self.test_student1, self.test_student2, self.test_norm, \
            self.test_norm1, self.test_norm2, self.a_j, self.b_j, self.beta_temp, self.mu_student, \
            self.v_student, self.v_studen1, self.v_student2, self.nu_student, self.nu_student1, self.nu_student2, \
            self.v_final, self.v_final1, self.v_final2 = \
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

        self.logs_output = theano.function([self.beta, self.y], [self.logZ, self.logZ1, self.logZ2, self.test_student,
                                                                 self.test_student1, self.test_student2, self.test_norm,
                                                                 self.test_norm1, self.test_norm2, self.a_j, self.b_j,
                                                                 self.beta_temp, self.mu_student,
                                                                 self.v_student, self.v_studen1, self.v_student2,
                                                                 self.nu_student, self.nu_student1, self.nu_student2,
                                                                 self.v_final, self.v_final1, self.v_final2])

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
            w, m, v = self.predict_probabilistic(Y[i, :])
            logZ, logZ1, logZ2, test_student, test_student1, test_student2, test_norm, test_norm1, test_norm2, \
                a_j, b_j, beta_tmp, mu_student, v_student, v_student1, v_student2, \
                nu_student, nu_student1, nu_student2, v_final, v_final1, v_final2 = self.logs_output(Beta[i, :], Y[i, :])

            if not np.all(a_j > 0):
                print("permutation {}. a_j is not positive".format(i))

            if not np.all(np.isfinite(v_final)):
                print("permutation {}. v_final is not finite".format(i))

            if not np.all(np.isfinite(beta_tmp)):
                print("permutation {}. beta_tmp is not finite".format(i))

            if not np.all(np.isfinite(mu_student)):
                print("permutation {}. mu_student is not finite".format(i))

            if not np.all(np.isfinite(v_student2)):
                print("permutation {}. v_student2 is not finite".format(i))

            if not np.all(np.isfinite(nu_student2)):
                print("permutation {}. nu_student2 is not finite".format(i))

            if not np.all(np.isfinite(test_student)):
                print("permutation {}. test_student is not finite".format(i))

            if not np.all(np.isfinite(test_student1)):
                print("permutation {}. test_student1 is not finite".format(i))

            if not np.all(np.isfinite(test_student2)):
                print("permutation {}. test_student2 is not finite".format(i))

            if not np.all(np.isfinite(test_norm)):
                print("permutation {}. test_norm is not finite".format(i))

            if not np.all(np.isfinite(test_norm1)):
                print("permutation {}. test_norm1 is not finite".format(i))

            if not np.all(np.isfinite(test_norm2)):
                print("permutation {}. test_norm2 is not finite".format(i))

            if not np.all(np.isfinite(logZ)):
                print("permutation {}. logZ is not finite".format(i))

            if not np.all(np.isfinite(logZ1)):
                print("permutation {}. logZ1 is not finite".format(i))

            if not np.all(np.isfinite(logZ2)):
                print("permutation {}. logZ2 is not finite".format(i))

            if not np.all(np.isfinite(a_j)):
                print("permutation {}. a_j is not finite".format(i))

            if not np.all(np.isfinite(b_j)):
                print("permutation {}. b_j is not finite".format(i))

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

    def sample_mean_ws(self):

        self.network.sample_mean_ws()
