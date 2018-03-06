import numpy as np
import theano
import six.moves.cPickle as pickle

from PBP_net_lista.BayesianListaHandler import BayesianListaHandler
from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.net_lista import net_lista
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler
from PBP_net_lista_single_matrices.net_lista import net_lista as net_list_single_matrices
from experiments.synthetic.experiments_parameters import load_synthetic_experiment_2
from freqLISTA.FrequentistListaHandler import FrequentistListaHandler
from freqLISTA.run_lista import sgd_optimization_lista, predict

import matplotlib.pyplot as plt


class SequentialComparator(object):
    def __init__(self, D, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1,
                 train_freq=True, train_bayes=True, train_shared_bayes=True, save_history=False):

        self.data_generator = DataGenerator(D, K)

        self.train_freq = train_freq
        self.train_bayes = train_bayes
        self.train_shared_bayes = train_shared_bayes

        self.save_history = save_history

        if self.train_freq:
            self.freq_lista = FrequentistListaHandler(D=D, K=K, L=L, X=self.data_generator.X,
                                                      learning_rate=learning_rate)
            self.freq_train_loss = []
            self.freq_validation_loss = []
            self.freq_train_f_meas = []
            self.freq_validation_f_meas = []

            if self.save_history:
                self.freq_w_hist = []
                self.freq_s_hist = []

        if self.train_bayes:
            self.bayesian_lista = BayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            self.bayesian_train_loss = []
            self.bayesian_validation_loss = []
            self.bayesian_train_f_meas = []
            self.bayesian_validation_f_meas = []

        if self.train_shared_bayes:
            self.shared_bayesian_lista = SingleBayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            self.shared_bayesian_train_loss = []
            self.shared_bayesian_validation_loss = []
            self.shared_bayesian_train_f_meas = []
            self.shared_bayesian_validation_f_meas = []

            if self.save_history:
                self.shared_bayes_w_hist = []
                self.shared_bayes_w_var_hist = []
                self.shared_bayes_s_hist = []
                self.shared_bayes_s_var_hist = []

        self.beta_train, self.y_train, _ = self.data_generator.new_sample(n_train_sample)
        self.beta_validation, self.y_validation, _ = self.data_generator.new_sample(n_validation_sample)

    def train_iteration(self):
        if self.train_freq:
            cur_freq_train_loss, cur_freq_train_f_meas = \
                self.freq_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train)
            self.freq_train_loss.append(cur_freq_train_loss)
            self.freq_train_f_meas.append(cur_freq_train_f_meas)

            if self.save_history:
                self.freq_w_hist.append(self.freq_lista.lista.W.get_value())
                self.freq_s_hist.append(self.freq_lista.lista.S.get_value())
        if self.train_bayes:
            cur_bayes_train_loss, cur_bayes_train_f_meas = \
                self.bayesian_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train)
            self.bayesian_train_loss.append(cur_bayes_train_loss)
            self.bayesian_train_f_meas.append(cur_bayes_train_f_meas)
        if self.train_shared_bayes:
            cur_sh_bayes_train_loss, cur_sh_bayes_train_f_meas = \
                self.shared_bayesian_lista.train_iteration(beta_train=self.beta_train, y_train=self.y_train)
            self.shared_bayesian_train_loss.append(cur_sh_bayes_train_loss)
            self.shared_bayesian_train_f_meas.append(cur_sh_bayes_train_f_meas)

            if self.save_history:
                self.shared_bayes_w_hist.append(
                    self.shared_bayesian_lista.pbp_instance.network.params_W_M.get_value())
                self.shared_bayes_w_var_hist.append(
                    self.shared_bayesian_lista.pbp_instance.network.params_W_V.get_value())
                self.shared_bayes_s_hist.append(
                    self.shared_bayesian_lista.pbp_instance.network.params_S_M.get_value())
                self.shared_bayes_s_var_hist.append(
                    self.shared_bayesian_lista.pbp_instance.network.params_S_V.get_value())

        if self.train_freq:
            cur_freq_valid_loss, cur_freq_valid_f_meas = \
                self.freq_lista.test(beta_test=self.beta_validation, y_test=self.y_validation)
            self.freq_validation_loss.append(cur_freq_valid_loss)
            self.freq_validation_f_meas.append(cur_freq_valid_f_meas)
        if self.train_bayes:
            cur_bayes_valid_loss, cur_bayes_valid_f_meas = \
                self.bayesian_lista.test(beta_test=self.beta_validation, y_test=self.y_validation)
            self.bayesian_validation_loss.append(cur_bayes_valid_loss)
            self.bayesian_validation_f_meas.append(cur_bayes_valid_f_meas)
        if self.train_shared_bayes:
            cur_sh_bayes_valid_loss, cur_sh_bayes_valid_f_meas = \
                self.shared_bayesian_lista.test(beta_test=self.beta_validation, y_test=self.y_validation)
            self.shared_bayesian_validation_loss.append(cur_sh_bayes_valid_loss)
            self.shared_bayesian_validation_f_meas.append(cur_sh_bayes_valid_f_meas)

    def plot_quality_history(self):
        if self.train_freq:
            plt.semilogy(self.freq_train_loss, label="freq train")
            plt.semilogy(self.freq_validation_loss, label="freq valid")
        if self.train_bayes:
            plt.semilogy(self.bayesian_train_loss, label="bayes train")
            plt.semilogy(self.bayesian_validation_loss, label="bayes valid")
        if self.train_shared_bayes:
            plt.semilogy(self.shared_bayesian_train_loss, label="shared bayes train")
            plt.semilogy(self.shared_bayesian_validation_loss, label="shared bayes valid")

        plt.legend()
        plt.title('NMSE')
        plt.show()

        if self.train_freq:
            plt.plot(self.freq_train_f_meas, label="freq train")
            plt.plot(self.freq_validation_f_meas, label="freq valid")
        if self.train_bayes:
            plt.plot(self.bayesian_train_f_meas, label="bayes train")
            plt.plot(self.bayesian_validation_f_meas, label="bayes valid")
        if self.train_shared_bayes:
            plt.plot(self.shared_bayesian_train_f_meas, label="shared bayes train")
            plt.plot(self.shared_bayesian_validation_f_meas, label="shared bayes valid")

        plt.legend()
        plt.title('F measure')
        plt.show()

if __name__ == '__main__':

    rseed, D, K, L, batch_size, validation_size, n_iter = load_synthetic_experiment_2()
    np.random.seed(rseed)

    saved_comparator_file_name = []#'best_model_bayes_lista_single_matrices.pkl'

    if not saved_comparator_file_name:
        comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size, n_validation_sample=validation_size)
    else:
        comparator = pickle.load(open(saved_comparator_file_name, 'rb'))

    for _ in range(n_iter):
        comparator.train_iteration()

    comparator.plot_quality_history()


    with open('cur.pkl', 'wb') as f:
        pickle.dump(comparator, f)

# D = 784, K = 100, L = 4, batch_size = 1000, validation_size = 100 - at the beginning bayes loss less than freg