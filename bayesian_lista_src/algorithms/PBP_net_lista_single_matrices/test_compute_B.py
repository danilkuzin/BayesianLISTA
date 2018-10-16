from unittest import TestCase
from PBP_net_lista.network_layer import compute_B

import theano
import theano.tensor as T
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class TestNetwork_layer(TestCase):
    def test_compute_B_theano_vs_np(self):
        self.W_M = np.eye(2)
        self.W_V = np.eye(2)
        self.y = np.ones((2,))
        y = T.vector()
        W_M = T.matrix()
        W_V = T.matrix()

        B_m, B_v = compute_B(W_M, W_V, y)

        f = theano.function([W_M, W_V, y], [B_m, B_v])

        theano_B_m, theano_B_v = f(self.W_M, self.W_V, self.y)
        numpy_B_m = np.dot(self.W_M, self.y)
        numpy_B_v = np.dot(self.W_V, self.y ** 2)
        cmp = (np.all(theano_B_m == numpy_B_m) and np.all(theano_B_v == numpy_B_v))
        self.assertTrue(np.all(cmp))


    def test_compute_B_distr_scalar(self):
        np_W_M = 1
        np_W_V = 1.5
        sample_size = 10000

        sample_W = np.random.normal(np_W_M, np.sqrt(np_W_V), (sample_size,))

        np_y = 8.5

        y = T.vector()
        W_M = T.matrix()
        W_V = T.matrix()

        B_m, B_v = compute_B(W_M, W_V, y)

        f = theano.function([W_M, W_V, y], [B_m, B_v])

        theano_B_m, theano_B_v = f([[np_W_M]], [[np_W_V]], [np_y])

        true_B_sample = sample_W * np_y

        est_B_sample = np.random.normal(theano_B_m, np.sqrt(theano_B_v), (sample_size,))

        sns.distplot(true_B_sample)
        sns.distplot(est_B_sample)
        plt.title('compute B scalar')
        plt.show()


    def test_compute_B_distr_vector(self):
        K = 3
        np_W_M = 2 * np.ones((K,))
        np_W_V = 3 * np.ones((K,))
        sample_size = 10000

        sample_W = np.zeros((sample_size, K), dtype=np.float64)
        for i in range(sample_size):
            sample_W[i, :] = np.random.normal(np_W_M, np.sqrt(np_W_V))

        np_y = np.random.randint(10, size=(K,)).astype(np.float64)

        y = T.vector()
        W_M = T.matrix()
        W_V = T.matrix()

        B_m, B_v = compute_B(W_M, W_V, y)

        f = theano.function([W_M, W_V, y], [B_m, B_v])

        theano_B_m, theano_B_v = f([np_W_M], [np_W_V], np_y)

        true_B_sample = np.dot(sample_W, np_y)

        est_B_sample = np.random.normal(theano_B_m, np.sqrt(theano_B_v), (sample_size,))

        sns.distplot(true_B_sample)
        sns.distplot(est_B_sample)
        plt.title('compute B vector')
        plt.show()


    def test_compute_B_distr_matrix(self):
        K = 10
        D = 3
        np_W_M = 2 * np.ones((D, K))
        np_W_V = 3 * np.ones((D, K))
        sample_size = 10000

        sample_W = np.zeros((sample_size, D, K), dtype=np.float64)
        for i in range(sample_size):
            sample_W[i, :, :] = np.random.normal(np_W_M, np.sqrt(np_W_V))

        np_y = np.random.randint(10, size=(K,)).astype(np.float64)

        y = T.vector()
        W_M = T.matrix()
        W_V = T.matrix()

        B_m, B_v = compute_B(W_M, W_V, y)

        f = theano.function([W_M, W_V, y], [B_m, B_v])

        theano_B_m, theano_B_v = f(np_W_M, np_W_V, np_y)

        true_B_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            true_B_sample[i, :] = np.dot(sample_W[i, :, :], np_y)

        est_B_sample = np.zeros((sample_size, D), dtype=np.float64)
        for i in range(sample_size):
            est_B_sample[i, :] = np.random.normal(theano_B_m, np.sqrt(theano_B_v))

        for d in range(D):
            sns.distplot(true_B_sample[:, d])
            sns.distplot(est_B_sample[:, d])
            plt.title('compute B matrix for column %d' %d)
            plt.show()





