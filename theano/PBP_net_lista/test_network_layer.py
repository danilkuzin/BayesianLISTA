from unittest import TestCase
from PBP_net_lista.network_layer import Network_layer

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def random_spike_and_slab(mean, std, omega_prob):
    omega = np.random.binomial(np.ones_like(omega_prob).astype(np.int), omega_prob)

    beta = np.random.normal(mean, std) * (np.ones_like(omega) - omega)

    return beta


def soft_threshold(v, thr_lambda):
    return np.sign(v) * np.maximum(abs(v) - thr_lambda, np.zeros_like(v))


class TestNetwork_layer(TestCase):
    def setUp(self):

        theano.config.exception_verbosity = 'high'
        theano.config.optimizer = 'None'

        self.K = 5
        self.D = 10

        self.W_M = 2 * np.ones((self.D, self.K))
        self.W_V = 3 * np.ones((self.D, self.K))
        self.S_M = np.random.randint(5, 10, size=(self.D, self.D)).astype(np.float64)
        self.S_V = 3 * np.random.rand(self.D, self.D)
        self.sample_size = 10000
        self.np_y = np.random.randint(10, size=(self.K,)).astype(np.float64)

        self.z_m = 3 * np.ones((self.D,), dtype=np.float64)
        self.z_v = 1.5 * np.ones((self.D,), dtype=np.float64)
        self.z_omega = np.random.rand(self.D)

        self.thr_lambda = 10

        self.layer = Network_layer(self.W_M, self.W_V, self.S_M, self.S_V, self.thr_lambda)

        self.plot = True

    # def test_compute_B(self):
    #
    #     sample_W = np.zeros((self.sample_size, self.D, self.K), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         sample_W[i, :, :] = np.random.normal(self.W_M, np.sqrt(self.W_V))
    #
    #     y = T.vector()
    #
    #     compute_B_func = theano.function([y], self.layer.compute_B(y))
    #     theano_B_m, theano_B_v = compute_B_func(self.np_y)
    #
    #     true_B_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         true_B_sample[i, :] = np.dot(sample_W[i, :, :], self.np_y)
    #
    #     est_B_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         est_B_sample[i, :] = np.random.normal(theano_B_m, np.sqrt(theano_B_v))
    #
    #     if self.plot:
    #         for d in range(self.D):
    #             sns.distplot(true_B_sample[:, d])
    #             sns.distplot(est_B_sample[:, d])
    #             plt.title('compute B matrix for column %d' % d)
    #             plt.show()

    # def test_compute_D(self):
    #
    #     sample_z = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         sample_z[i, :] = random_spike_and_slab(self.z_m, np.sqrt(self.z_v), self.z_omega)
    #
    #     sample_S = np.zeros((self.sample_size, self.D, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         sample_S[i, :, :] = np.random.normal(self.S_M, np.sqrt(self.S_V))
    #
    #     z_m = T.vector()
    #     z_v = T.vector()
    #     z_omega = T.vector()
    #
    #     compute_D_func = theano.function([z_omega, z_m, z_v], self.layer.compute_D(z_omega, z_m, z_v))
    #     theano_D_m, theano_D_v = compute_D_func(self.z_omega, self.z_m, self.z_v)
    #
    #     true_D_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         true_D_sample[i, :] = np.dot(sample_S[i, :, :], sample_z[i, :])
    #
    #     est_D_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         est_D_sample[i, :] = np.random.normal(theano_D_m, np.sqrt(theano_D_v))
    #
    #     if self.plot:
    #         d = 5
    #         sns.distplot(true_D_sample[:, d], label='true')
    #         sns.distplot(est_D_sample[:, d], label='estimated')
    #         plt.legend()
    #         plt.title('compute D matrix for column %d' % d)
    #         plt.savefig('d_testing.eps', format='eps')


    # def test_compute_C(self):
    #
    #     y = T.vector()
    #
    #     z_m = T.vector()
    #     z_v = T.vector()
    #     z_omega = T.vector()
    #
    #     compute_B_func = theano.function([y], self.layer.compute_B(y))
    #     np_B_m, np_B_v = compute_B_func(self.np_y)
    #
    #     compute_D_func = theano.function([z_omega, z_m, z_v], self.layer.compute_D(z_omega, z_m, z_v))
    #     np_D_m, np_D_v = compute_D_func(self.z_omega, self.z_m, self.z_v)
    #
    #     B_m = T.vector()
    #     B_v = T.vector()
    #
    #     D_m = T.vector()
    #     D_v = T.vector()
    #
    #     sample_B = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         sample_B[i, :] = np.random.normal(np_B_m, np.sqrt(np_B_v))
    #
    #     sample_D = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         sample_D[i, :] = np.random.normal(np_D_m, np.sqrt(np_D_v))
    #
    #     compute_C_func = theano.function([B_m, B_v, D_m, D_v], self.layer.compute_C(B_m, B_v, D_m, D_v))
    #     theano_C_m, theano_C_v = compute_C_func(np_B_m, np_B_v, np_D_m, np_D_v)
    #
    #     true_C_sample = sample_B + sample_D
    #
    #     est_C_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
    #     for i in range(self.sample_size):
    #         est_C_sample[i, :] = np.random.normal(theano_C_m, np.sqrt(theano_C_v))
    #
    #     if self.plot:
    #         d = 5
    #         sns.distplot(true_C_sample[:, d], label='true')
    #         sns.distplot(est_C_sample[:, d], label='estimated')
    #         plt.legend()
    #         plt.title('compute C vector for column %d' % d)
    #         plt.show()

    def test_compute_new_z(self):
        y = T.vector()

        z_m = T.vector()
        z_v = T.vector()
        z_omega = T.vector()

        # compute_B_func = theano.function([y], self.layer.compute_B(y))
        # np_B_m, np_B_v = compute_B_func(self.np_y)
        #
        # compute_D_func = theano.function([z_omega, z_m, z_v], self.layer.compute_D(z_omega, z_m, z_v))
        # np_D_m, np_D_v = compute_D_func(self.z_omega, self.z_m, self.z_v)
        #
        # B_m = T.vector()
        # B_v = T.vector()
        #
        # D_m = T.vector()
        # D_v = T.vector()
        #
        # sample_B = np.zeros((self.sample_size, self.D), dtype=np.float64)
        # for i in range(self.sample_size):
        #     sample_B[i, :] = np.random.normal(np_B_m, np.sqrt(np_B_v))
        #
        # sample_D = np.zeros((self.sample_size, self.D), dtype=np.float64)
        # for i in range(self.sample_size):
        #     sample_D[i, :] = np.random.normal(np_D_m, np.sqrt(np_D_v))
        #
        # compute_C_func = theano.function([B_m, B_v, D_m, D_v], self.layer.compute_C(B_m, B_v, D_m, D_v))
        # np_C_m, np_C_v = compute_C_func(np_B_m, np_B_v, np_D_m, np_D_v)


        np_C_m = -2 * np.ones(self.D, dtype=np.float64)
        np_C_v = 1 * np.ones(self.D, dtype=np.float64)

        sample_C = np.zeros((self.sample_size, self.D), dtype=np.float64)
        for i in range(self.sample_size):
            sample_C[i, :] = np.random.normal(np_C_m, np.sqrt(np_C_v))

        C_m = T.vector()
        C_v = T.vector()

        self.thr_lambda = 1

        self.layer = Network_layer(self.W_M, self.W_V, self.S_M, self.S_V, self.thr_lambda)

        compute_z_func = theano.function([C_m, C_v], self.layer.compute_new_z(C_m, C_v))
        theano_z_omega, theano_z_m, theano_z_v = compute_z_func(np_C_m, np_C_v)

        true_z_sample = soft_threshold(sample_C, self.thr_lambda)

        est_z_sample = np.zeros((self.sample_size, self.D), dtype=np.float64)
        for i in range(self.sample_size):
            est_z_sample[i, :] = random_spike_and_slab(theano_z_m, np.sqrt(theano_z_v), theano_z_omega)

        if self.plot:
            d = 3
            sns.distplot(true_z_sample[:, d], label='true')
            sns.distplot(est_z_sample[:, d], label='estimated')
            plt.legend()
            plt.title('compute z_new vector for column %d' % d)
            plt.savefig('z_new_testing.eps', format='eps')

    # def test_output_probabilistic(self):
    #
    #     y = T.vector()
    #     z_m = T.vector()
    #     z_v = T.vector()
    #     z_omega = T.vector()
    #
    #     fprop_func = theano.function([z_omega, z_m, z_v, y], self.layer.output_probabilistic(z_omega, z_m, z_v, y))
    #     z_omega_new, z_m_new, z_v_new = fprop_func(self.z_omega, self.z_m, self.z_v, self.np_y)
    #     print('z_omega_new:{0}\n, z_m_new:{1}\n, z_v_new:{2}'.format(z_omega_new, z_m_new, z_v_new))
    #
    # def test_output_deterministic(self):
    #     y = T.vector()
    #     out_prev = T.vector()
    #
    #     out_prev_np = 2 * np.ones((self.D))
    #
    #     fprop_func = theano.function([out_prev, y], self.layer.output_deterministic(out_prev, y))
    #     out_cur = fprop_func(out_prev_np, self.np_y)
    #     print('out_cur:{}'.format(out_cur))
