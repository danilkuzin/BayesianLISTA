from tqdm import trange

from tf.data.synthetic.data_generator import DataGenerator
from tf.algorithms.listapbp.handler import SingleBayesianListaHandler
from tf.data.mnist.mnist_data import MnistData
from tf.algorithms.lista.handler import ListaHandler
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import six.moves.cPickle as pickle


class ActiveLearningExperiments(object):
    def __init__(self, update_size):
        self.L = 8
        self.learning_rate = 0.0001
        self.n_train_iter = 200
        self.update_size = update_size

        self.freq_validation_loss = []
        self.freq_validation_f_meas = []

        self.active_bayesian_validation_loss = []
        self.active_bayesian_validation_f_meas = []

        self.non_active_shared_bayesian_validation_loss = []
        self.non_active_shared_bayesian_validation_f_meas = []

    def get_mnist_data_active_learning(self):
        self.K=100
        n_train, n_pool, n_test = 10, 1000, 100
        data = MnistData(K=self.K, train_size=n_train + n_pool, valid_size=n_test)
        data.check_download()
        data.random_dictionary(normalise=True)
        self.D = data.beta_train.shape[1]

        self.train_data_beta, self.train_data_y = data.beta_train[:n_train], data.y_train[:n_train]
        self.pool_data_beta, self.pool_data_y = data.beta_train[n_train+1:], data.y_train[n_train+1:]
        self.test_data_beta, self.test_data_y = data.beta_validation, data.y_validation
        self.active_pool_data_beta, self.active_pool_data_y = self.pool_data_beta, self.pool_data_y
        self.active_train_data_beta, self.active_train_data_y = self.train_data_beta, self.train_data_y

        self.X = data.X
        # return train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y


    def get_synthetic_data_active_learning(self):
        self.D = 100
        self.K = 50
        n_train, n_pool, n_test = 900, 500, 100
        data_generator = DataGenerator(self.D, self.K)
        self.train_data_beta, self.train_data_y, _ = data_generator.new_sample(n_train)
        self.pool_data_beta, self.pool_data_y, _ = data_generator.new_sample(n_pool)
        self.test_data_beta, self.test_data_y, _ = data_generator.new_sample(n_test)
        self.active_pool_data_beta, self.active_pool_data_y = self.pool_data_beta, self.pool_data_y
        self.active_train_data_beta, self.active_train_data_y = self.train_data_beta, self.train_data_y

        self.X = data_generator.X
        # return train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y

    def init_and_pretrain_lista(self):
        self.freq_lista = ListaHandler(D=self.D, K=self.K, L=self.L, X=self.X,
                                                  learning_rate=self.learning_rate, initial_lambda=0.1)
        self.active_bayesian_lista = SingleBayesianListaHandler(D=self.D, K=self.K, L=self.L, X=self.X, initial_lambda=0.1)
        self.non_active_bayesian_lista = SingleBayesianListaHandler(D=self.D, K=self.K, L=self.L, X=self.X, initial_lambda=0.1)

        for _ in trange(self.n_train_iter):
            self.learning_iter()

        self.update_quality()


    def choose_next_train_active_from_pool(self):
        w, m, v = self.active_bayesian_lista.predict_probabilistic(self.active_pool_data_y)
        spike_and_slab_var = (1-w) * (v + w * m**2)
        spike_and_slab_entropy = np.sum(spike_and_slab_var, axis=1)
        upd_ind = np.argpartition(spike_and_slab_entropy, -self.update_size)[-self.update_size:]

        self.active_train_data_beta = np.append(self.active_train_data_beta, self.active_pool_data_beta[upd_ind], axis=0)
        self.active_train_data_y = np.append(self.active_train_data_y, self.active_pool_data_y[upd_ind], axis=0)
        self.active_pool_data_beta = np.delete(self.active_pool_data_beta, upd_ind, axis=0)
        self.active_pool_data_y = np.delete(self.active_pool_data_y, upd_ind, axis=0)


    def choose_next_random_from_pool(self):
        upd_ind = np.random.choice(self.pool_data_beta.shape[0], size=self.update_size, replace=False)

        self.train_data_beta = np.append(self.train_data_beta, self.pool_data_beta[upd_ind], axis=0)
        self.train_data_y = np.append(self.train_data_y, self.pool_data_y[upd_ind], axis=0)
        self.pool_data_beta = np.delete(self.pool_data_beta, upd_ind, axis=0)
        self.pool_data_y = np.delete(self.pool_data_y, upd_ind, axis=0)

    def learning_iter(self):
        self.freq_lista.train_iteration(beta_train=self.train_data_beta, y_train=self.train_data_y)
        self.active_bayesian_lista.train_iteration(beta_train=self.active_train_data_beta,
                                                   y_train=self.active_train_data_y)
        self.non_active_bayesian_lista.train_iteration(beta_train=self.train_data_beta,
                                                       y_train=self.train_data_y)

    def update_quality(self):
        cur_freq_valid_loss, cur_freq_valid_f_meas = \
            self.freq_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        self.freq_validation_loss.append(cur_freq_valid_loss)
        self.freq_validation_f_meas.append(cur_freq_valid_f_meas)

        cur_active_bayes_valid_loss, cur_active_bayes_valid_f_meas = \
            self.active_bayesian_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        self.active_bayesian_validation_loss.append(cur_active_bayes_valid_loss)
        self.active_bayesian_validation_f_meas.append(cur_active_bayes_valid_f_meas)

        cur_non_active_bayes_valid_loss, cur_non_active_bayes_valid_f_meas = \
            self.non_active_bayesian_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        self.non_active_shared_bayesian_validation_loss.append(cur_non_active_bayes_valid_loss)
        self.non_active_shared_bayesian_validation_f_meas.append(cur_non_active_bayes_valid_f_meas)

    def plot(self):
        plt.plot(self.freq_validation_loss, label='freq')
        plt.plot(self.non_active_shared_bayesian_validation_loss, label='nonactive')
        plt.plot(self.active_bayesian_validation_loss, label='active')
        plt.legend()
        plt.show()
