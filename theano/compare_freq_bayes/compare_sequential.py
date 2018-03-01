import numpy as np
import theano
import six.moves.cPickle as pickle

from PBP_net_lista.BayesianListaHandler import BayesianListaHandler
from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.net_lista import net_lista
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler
from PBP_net_lista_single_matrices.net_lista import net_lista as net_list_single_matrices
from freqLISTA.FrequentistListaHandler import FrequentistListaHandler
from freqLISTA.run_lista import sgd_optimization_lista, predict

import matplotlib.pyplot as plt


class SequentialComparator(object):
    def __init__(self, D, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1):

        self.data_generator = DataGenerator(D, K)

        self.freq_lista = FrequentistListaHandler(D=D, K=K, L=L, X=self.data_generator.X, learning_rate=learning_rate)
        self.bayesian_lista = BayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)
        self.shared_bayesian_lista = SingleBayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)

        self.freq_train_loss = []
        self.bayesian_train_loss = []
        self.shared_bayesian_train_loss = []

        self.freq_validation_loss = []
        self.bayesian_validation_loss = []
        self.shared_bayesian_validation_loss = []

        self.beta_train, self.y_train, _ = self.data_generator.new_sample(n_train_sample)
        self.beta_validation, self.y_validation, _ = self.data_generator.new_sample(n_validation_sample)

    def train_iteration(self):

        self.freq_train_loss.append(
            self.freq_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train))
        self.bayesian_train_loss.append(
            self.bayesian_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train))
        self.shared_bayesian_train_loss.append(
            self.shared_bayesian_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train))

        self.freq_validation_loss.append(
            self.freq_lista.test(beta_test=self.beta_validation, y_test=self.y_validation))
        self.bayesian_validation_loss.append(
            self.bayesian_lista.test(beta_test=self.beta_validation, y_test=self.y_validation))
        self.shared_bayesian_validation_loss.append(
            self.shared_bayesian_lista.test(beta_test=self.beta_validation, y_test=self.y_validation))


if __name__ == '__main__':

    np.random.seed(1)

    D = 100
    K = 50
    L = 4

    batch_size = 5000
    validation_size = 100

    comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size, n_validation_sample=validation_size)

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