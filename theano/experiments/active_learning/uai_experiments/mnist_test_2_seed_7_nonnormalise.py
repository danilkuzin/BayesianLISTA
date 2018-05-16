from tqdm import trange

from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler
from compare_mnist.mnist_data import MnistData
from freqLISTA.FrequentistListaHandler import FrequentistListaHandler
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import six.moves.cPickle as pickle

class ActiveLearningExperiments(object):
    def __init__(self, update_size):
        self.L = 20
        self.learning_rate = 0.0001
        self.n_train_iter = 50
        self.update_size = update_size

        self.freq_validation_loss = []
        self.freq_validation_f_meas = []

        self.active_bayesian_validation_loss = []
        self.active_bayesian_validation_f_meas = []

        self.non_active_shared_bayesian_validation_loss = []
        self.non_active_shared_bayesian_validation_f_meas = []

    def get_mnist_data_active_learning(self):
        self.K=250
        n_train, n_pool, n_test = 50, 500, 100
        data = MnistData(K=self.K, train_size=n_train + n_pool, valid_size=n_test)
        data.check_download()
        data.learn_dictionary()
        self.D = data.train_data.shape[1]

        self.train_data_beta, self.train_data_y = data.train_data[:n_train], data.y_train[:n_train]
        self.pool_data_beta, self.pool_data_y = data.train_data[n_train+1:], data.y_train[n_train+1:]
        self.test_data_beta, self.test_data_y = data.validation_data, data.y_validation
        self.active_pool_data_beta, self.active_pool_data_y = self.pool_data_beta, self.pool_data_y
        self.active_train_data_beta, self.active_train_data_y = self.train_data_beta, self.train_data_y

        self.X = data.X
        # return train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y


    def get_synthetic_data_active_learning(self):
        self.D = 100
        self.K = 50
        n_train, n_pool, n_test = 500, 1000, 100
        data_generator = DataGenerator(self.D, self.K)
        self.train_data_beta, self.train_data_y, _ = data_generator.new_sample(n_train)
        self.pool_data_beta, self.pool_data_y, _ = data_generator.new_sample(n_pool)
        self.test_data_beta, self.test_data_y, _ = data_generator.new_sample(n_test)
        self.active_pool_data_beta, self.active_pool_data_y = self.pool_data_beta, self.pool_data_y
        self.active_train_data_beta, self.active_train_data_y = self.train_data_beta, self.train_data_y

        self.X = data_generator.X
        # return train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y

    def init_and_pretrain_lista(self):
        #self.freq_lista = FrequentistListaHandler(D=self.D, K=self.K, L=self.L, X=self.X,
        #                                          learning_rate=self.learning_rate)
        self.active_bayesian_lista = SingleBayesianListaHandler(D=self.D, K=self.K, L=self.L, X=self.X)
        self.non_active_bayesian_lista = SingleBayesianListaHandler(D=self.D, K=self.K, L=self.L, X=self.X)

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
        #self.freq_lista.train_iteration(beta_train=self.train_data_beta, y_train=self.train_data_y)
        self.active_bayesian_lista.train_iteration(beta_train=self.active_train_data_beta,
                                                   y_train=self.active_train_data_y)
        self.non_active_bayesian_lista.train_iteration(beta_train=self.train_data_beta,
                                                       y_train=self.train_data_y)

    def update_quality(self):
        # cur_freq_valid_loss, cur_freq_valid_f_meas = \
        #     self.freq_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        # self.freq_validation_loss.append(cur_freq_valid_loss)
        # self.freq_validation_f_meas.append(cur_freq_valid_f_meas)

        cur_active_bayes_valid_loss, cur_active_bayes_valid_f_meas = \
            self.active_bayesian_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        self.active_bayesian_validation_loss.append(cur_active_bayes_valid_loss)
        self.active_bayesian_validation_f_meas.append(cur_active_bayes_valid_f_meas)

        cur_non_active_bayes_valid_loss, cur_non_active_bayes_valid_f_meas = \
            self.non_active_bayesian_lista.test(beta_test=self.test_data_beta, y_test=self.test_data_y)
        self.non_active_shared_bayesian_validation_loss.append(cur_non_active_bayes_valid_loss)
        self.non_active_shared_bayesian_validation_f_meas.append(cur_non_active_bayes_valid_f_meas)

    def plot(self):
        #plt.plot(active_learning_experiments.freq_validation_loss, label='freq')
        plt.plot(active_learning_experiments.non_active_shared_bayesian_validation_loss, label='nonactive')
        plt.plot(active_learning_experiments.active_bayesian_validation_loss, label='active')
        plt.legend()
        plt.show()

if __name__=='__main__':

    # #for rseed in trange(10):
    # rseed = 1
    # np.random.seed(rseed)
    # tf.set_random_seed(rseed)
    #
    # n_upd_iter = 5
    # active_learning_experiments = ActiveLearningExperiments(update_size=10)
    # #active_learning_experiments.get_synthetic_data_active_learning()
    # active_learning_experiments.get_mnist_data_active_learning()
    # active_learning_experiments.init_and_pretrain_lista()
    #
    # for i in trange(n_upd_iter):
    #     active_learning_experiments.choose_next_random_from_pool()
    #     active_learning_experiments.choose_next_train_active_from_pool()
    #     for j in range(100):
    #         active_learning_experiments.learning_iter()
    #     active_learning_experiments.update_quality()
    #
    # active_learning_experiments.plot()

    rseed = 7
    np.random.seed(rseed)
    tf.set_random_seed(rseed)

    saved_file_name = []#'synthetic_D_100_K_50_L_8_200_iter_train_900_train_size.pkl'

    n_upd_iter = 10
    update_size = 1

    if not saved_file_name:
        active_learning_experiments = ActiveLearningExperiments(update_size=update_size)
        active_learning_experiments.get_mnist_data_active_learning()
        active_learning_experiments.init_and_pretrain_lista()
    else:
        active_learning_experiments = pickle.load(open(saved_file_name, 'rb'))

    #    with open('synthetic_D_100_K_50_L_8_200_iter_train_900_train_size.pkl', 'wb') as f:
    #        pickle.dump(active_learning_experiments, f)


    for i in trange(n_upd_iter):
        active_learning_experiments.choose_next_random_from_pool()
        active_learning_experiments.choose_next_train_active_from_pool()
        for j in range(10):
            active_learning_experiments.learning_iter()
        active_learning_experiments.update_quality()

    active_learning_experiments.plot()

    freq_validation_loss = active_learning_experiments.freq_validation_loss
    non_active_bayes_validation_loss = active_learning_experiments.non_active_shared_bayesian_validation_loss
    active_bayes_validation_loss = active_learning_experiments.active_bayesian_validation_loss
    freq_validation_f_measure = active_learning_experiments.freq_validation_f_meas
    non_active_bayes_validation_f_measure = active_learning_experiments.non_active_shared_bayesian_validation_f_meas
    active_bayes_validation_f_measure = active_learning_experiments.active_bayesian_validation_f_meas
    np.savez('mnist_active_rseed_{}'.format(rseed), freq_validation_loss=freq_validation_loss,
             non_active_bayes_validation_loss=non_active_bayes_validation_loss,
             active_bayes_validation_loss=active_bayes_validation_loss,
             freq_validation_f_measure=freq_validation_f_measure,
             non_active_bayes_validation_f_measure=non_active_bayes_validation_f_measure,
             active_bayes_validation_f_measure=active_bayes_validation_f_measure)
