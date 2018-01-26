from unittest import TestCase
from PBP_net_lista.network_layer import compute_D

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# def random_spike_and_slab(mean, std, omega_prob):
#     omega = np.random.binomial(np.ones_like(omega_prob).astype(np.int), omega_prob)
#
#     beta = np.random.normal(mean, std) * (np.ones_like(omega) - omega)
#
#     return beta


def compute_D_normal(S_M, S_V, z_m_prev, z_v_prev):
    D_m = T.dot(S_M, z_m_prev)
    D_v = T.dot(S_M ** 2, z_v_prev) + T.dot(S_V, z_m_prev ** 2) + \
          T.dot(S_V, z_v_prev)
    return D_m, D_v


class TestNetwork_layer(TestCase):

    def test_compute_D(self):
        np.random.seed(1000)
        D = 10
        np_z_m = 3 * np.ones((D,), dtype=np.float64)
        np_z_v = 1.5 * np.ones((D,), dtype=np.float64)
        np_z_omega = np.random.rand(D)#0.8 * np.ones((D,), dtype=np.float64)#
        np_S_M = np.random.randint(5, 10, size=(D, D)).astype(np.float64)
        np_S_V = 3 * np.random.rand(D, D)

        sample_size = 10000

        sample_z = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            sample_z[i, :] = random_spike_and_slab(np_z_m, np.sqrt(np_z_v), np_z_omega)

        sample_S = np.zeros((sample_size, D, D), dtype=np.float64)
        for i in range(sample_size):
            sample_S[i, :, :] = np.random.normal(np_S_M, np.sqrt(np_S_V))

        z_m = T.vector()
        z_v = T.vector()
        z_omega = T.vector()

        S_M = T.matrix()
        S_V = T.matrix()

        D_m, D_v = compute_D(S_M, S_V, z_omega, z_m, z_v)

        f = theano.function([S_M, S_V, z_omega, z_m, z_v], [D_m, D_v])

        theano_D_m, theano_D_v = f(np_S_M, np_S_V, np_z_omega, np_z_m, np_z_v)

        true_D_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            true_D_sample[i, :] = np.dot(sample_S[i, :, :], sample_z[i, :])


        est_D_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            est_D_sample[i, :] = np.random.normal(theano_D_m, np.sqrt(theano_D_v))

        d = 5
        sns.distplot(true_D_sample[:, d], label='true')
        sns.distplot(est_D_sample[:, d], label='estimated')
        plt.legend()
        plt.title('compute D matrix for column %d' %d)
        plt.show()


    # def test_compute_D_normal(self):
    #     np.random.seed(1000)
    #     D = 20
    #     np_z_m = 10 * np.ones((D,), dtype=np.float64)
    #     np_z_v = 1.5 * np.ones((D,), dtype=np.float64)
    #     np_S_M = np.random.randint(5, 10, size=(D, D)).astype(np.float64)
    #     np_S_V = 3 * np.random.rand(D, D)
    #
    #     sample_size = 10000
    #
    #     sample_z = np.zeros((sample_size, D), dtype=np.float64)
    #     for i in range(sample_size):
    #         sample_z[i, :] = np.random.normal(np_z_m, np.sqrt(np_z_v))
    #
    #     sample_S = np.zeros((sample_size, D, D), dtype=np.float64)
    #     for i in range(sample_size):
    #         sample_S[i, :, :] = np.random.normal(np_S_M, np.sqrt(np_S_V))
    #
    #     z_m = T.vector()
    #     z_v = T.vector()
    #
    #     S_M = T.matrix()
    #     S_V = T.matrix()
    #
    #     D_m, D_v = compute_D_normal(S_M, S_V, z_m, z_v)
    #
    #     f = theano.function([S_M, S_V, z_m, z_v], [D_m, D_v])
    #
    #     theano_D_m, theano_D_v = f(np_S_M, np_S_V, np_z_m, np_z_v)
    #
    #     true_D_sample = np.zeros((sample_size, D), dtype=np.float64)
    #     for i in range(sample_size):
    #         true_D_sample[i, :] = np.dot(sample_S[i, :, :], sample_z[i, :])
    #
    #     est_D_sample = np.zeros((sample_size, D), dtype=np.float64)
    #     for i in range(sample_size):
    #         est_D_sample[i, :] = np.random.normal(theano_D_m, np.sqrt(theano_D_v))
    #
    #     for d in range(D):
    #         sns.distplot(true_D_sample[:, d], label='true')
    #         sns.distplot(est_D_sample[:, d], label='estimated')
    #         plt.legend()
    #         plt.title('compute normal D matrix for column %d' % d)
    #         plt.show()






