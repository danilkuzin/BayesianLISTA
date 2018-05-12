from unittest import TestCase
from PBP_net_lista.network_layer import compute_new_z

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from PBP_net_lista.test_compute_D import random_spike_and_slab


def soft_threshold(v, thr_lambda):
    return np.sign(v) * np.maximum(abs(v) - thr_lambda, np.zeros_like(v))


class TestNetwork_layer(TestCase):

    def test_compute_new_z(self):
        np.random.seed(1000)
        D = 10
        np_C_m = -6.5 * np.ones(D, dtype=np.float64)
        np_C_v = 5 * np.ones(D, dtype=np.float64)
        np_lambda = 1

        sample_size = 100000

        sample_C = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            sample_C[i, :] = np.random.normal(np_C_m, np.sqrt(np_C_v))

        C_m = T.vector()
        C_v = T.vector()

        lambda_thr = T.scalar()

        z_m, z_v, z_omega = compute_new_z(C_m, C_v, lambda_thr)

        f = theano.function([C_m, C_v, lambda_thr], [z_m, z_v, z_omega])

        theano_z_m, theano_z_v, theano_z_omega = f(np_C_m, np_C_v, np_lambda)


        true_z_sample = soft_threshold(sample_C, np_lambda)


        est_z_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            est_z_sample[i, :] = random_spike_and_slab(theano_z_m, np.sqrt(theano_z_v), theano_z_omega)

        d = 3
        sns.distplot(true_z_sample[:, d], label='true')
        sns.distplot(est_z_sample[:, d], label='estimated')
        plt.legend()
        plt.title('compute z_new vector for column %d' %d)
        plt.show()









