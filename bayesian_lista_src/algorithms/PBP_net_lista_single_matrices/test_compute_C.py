from unittest import TestCase
from PBP_net_lista.network_layer import compute_C

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



class TestNetwork_layer(TestCase):

    def test_compute_C(self):
        np.random.seed(1000)
        D = 10
        np_B_m = 2.5 * np.random.rand(D)
        np_B_v = 1.5 * np.random.rand(D)
        np_D_m = 8 * np.random.rand(D)
        np_D_v = 31 * np.random.rand(D)

        sample_size = 10000

        sample_B = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            sample_B[i, :] = np.random.normal(np_B_m, np.sqrt(np_B_v))

        sample_D = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            sample_D[i, :] = np.random.normal(np_D_m, np.sqrt(np_D_v))

        B_m = T.vector()
        B_v = T.vector()

        D_m = T.vector()
        D_v = T.vector()

        C_m, C_v = compute_C(B_m, B_v, D_m, D_v)

        f = theano.function([B_m, B_v, D_m, D_v], [C_m, C_v])

        theano_C_m, theano_C_v = f(np_B_m, np_B_v, np_D_m, np_D_v)

        true_C_sample = sample_B + sample_D


        est_C_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            est_C_sample[i, :] = np.random.normal(theano_C_m, np.sqrt(theano_C_v))

        d = 5
        sns.distplot(true_C_sample[:, d], label='true')
        sns.distplot(est_C_sample[:, d], label='estimated')
        plt.legend()
        plt.title('compute C vector for column %d' %d)
        plt.show()









