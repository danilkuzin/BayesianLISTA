import numpy as np
import theano
import six.moves.cPickle as pickle

from PBP_net_lista.BayesianListaHandler import BayesianListaHandler
from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.net_lista import net_lista
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler
from PBP_net_lista_single_matrices.net_lista import net_lista as net_list_single_matrices
from compare_freq_bayes.compare_sequential import SequentialComparator
from compare_mnist.mnist_data import MnistData
from freqLISTA.FrequentistListaHandler import FrequentistListaHandler
from freqLISTA.run_lista import sgd_optimization_lista, predict

import matplotlib.pyplot as plt


class MnistSequentialComparator(object):
    def __init__(self, K, L, learning_rate=0.0001):

        self.data = MnistData(K=K)
        self.data.check_download()
        self.data.learn_dictionary()
        self.D = self.data.train_data.shape[1]

        self.freq_lista = FrequentistListaHandler(D=self.D, K=K, L=L, X=self.data.X, learning_rate=learning_rate)
        self.bayesian_lista = BayesianListaHandler(D=self.D, K=K, L=L, X=self.data.X)
        self.shared_bayesian_lista = SingleBayesianListaHandler(D=self.D, K=K, L=L, X=self.data.X)

        self.freq_train_loss = []
        self.bayesian_train_loss = []
        self.shared_bayesian_train_loss = []

        self.freq_validation_loss = []
        self.bayesian_validation_loss = []
        self.shared_bayesian_validation_loss = []

    def train_iteration(self):

        self.freq_train_loss.append(
            self.freq_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))
        self.bayesian_train_loss.append(
            self.bayesian_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))
        self.shared_bayesian_train_loss.append(
            self.shared_bayesian_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))

        self.freq_validation_loss.append(
            self.freq_lista.test(beta_test=self.data.validation_data, y_test=self.data.y_validation))
        self.bayesian_validation_loss.append(
            self.bayesian_lista.test(beta_test=self.data.validation_data, y_test=self.data.y_validation))
        self.shared_bayesian_validation_loss.append(
            self.shared_bayesian_lista.test(beta_test=self.data.validation_data, y_test=self.data.y_validation))


if __name__ == '__main__':

    np.random.seed(1)

    K = 100
    L = 4

    batch_size = 5000
    validation_size = 100

    comparator = MnistSequentialComparator(K, L, learning_rate=0.0001)

    n_iter = 500

    for _ in range(n_iter):
        comparator.train_iteration()

    plt.semilogy(comparator.freq_train_loss, label="freq train loss")
    plt.semilogy(comparator.bayesian_train_loss, label="bayes train loss")
    plt.semilogy(comparator.shared_bayesian_train_loss, label="shared bayes train loss")

    plt.semilogy(comparator.freq_validation_loss, label="freq valid loss")
    plt.semilogy(comparator.bayesian_validation_loss, label="bayes valid loss")
    plt.semilogy(comparator.shared_bayesian_validation_loss, label="shared bayes valid loss")

    plt.legend()
    plt.show()

# D = 100, K = 50, L = 4, batch_size = 1000 - at iterations 0-200 bayesian loss is smaller than freq