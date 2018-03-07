import numpy as np
import theano
import six.moves.cPickle as pickle

from PBP_net_lista.BayesianListaHandler import BayesianListaHandler
from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.net_lista import net_lista
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler
from PBP_net_lista_single_matrices.net_lista import net_lista as net_list_single_matrices
from experiments.synthetic.experiments_parameters import load_synthetic_experiment_2, load_quick_experiment
from fista.fIstaHandler import FistaHandler
from freqLISTA.FrequentistListaHandler import FrequentistListaHandler
from freqLISTA.run_lista import sgd_optimization_lista, predict

import matplotlib.pyplot as plt

from ista.IstaHandler import IstaHandler


class Recorder(object):
    def __init__(self, handler, save_history, name):
        self.handler = handler
        self.save_history = save_history
        self.name = name

        self.train_loss = []
        self.validation_loss = []
        self.train_f_meas = []
        self.validation_f_meas = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        cur_train_loss, cur_train_f_meas = \
            self.handler.train_iteration(beta_train=beta_train, y_train=y_train)
        self.train_loss.append(cur_train_loss)
        self.train_f_meas.append(cur_train_f_meas)

        cur_valid_loss, cur_valid_f_meas = \
            self.handler.test(beta_test=beta_validation, y_test=y_validation)
        self.validation_loss.append(cur_valid_loss)
        self.validation_f_meas.append(cur_valid_f_meas)

    def save_numpy(self, filename):
        np.savez('{}_{}'.format(filename, self.name), D=self.handler.D, K=self.handler.K, L=self.handler.L,
                 train_loss=self.train_loss, validation_loss=self.validation_loss,
                 train_f_meas=self.train_f_meas, validation_f_meas=self.validation_f_meas)


class FrequentistRecorder(Recorder):
    pass


class IstaRecorder(FrequentistRecorder):
    def __init__(self, handler, name):
        save_history = False
        super().__init__(handler, save_history, name)


class FistaRecorder(FrequentistRecorder):
    def __init__(self, handler, name):
        save_history = False
        super().__init__(handler, save_history, name)


class BayesianRecorder(Recorder):
    pass


class SeparateBayesianRecorder(BayesianRecorder):
    def __init__(self, handler, name):
        save_history = False
        super().__init__(handler, save_history, name)


class SharedBayesianRecorder(BayesianRecorder):
    def __init__(self, handler, save_history, name):
        super().__init__(handler, save_history, name)
        if self.save_history:
            self.w_hist = []
            self.w_var_hist = []
            self.s_hist = []
            self.s_var_hist = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        super().train_and_record(beta_train, y_train, beta_validation, y_validation)

        if self.save_history:
            self.w_hist.append(
                self.handler.pbp_instance.network.params_W_M.get_value())
            self.w_var_hist.append(
                self.handler.pbp_instance.network.params_W_V.get_value())
            self.s_hist.append(
                self.handler.pbp_instance.network.params_S_M.get_value())
            self.s_var_hist.append(
                self.handler.pbp_instance.network.params_S_V.get_value())

    def save_numpy(self, filename):
        bayes_W_M = self.handler.pbp_instance.network.params_W_M.get_value()
        bayes_W_V = self.handler.pbp_instance.network.params_W_V.get_value()
        bayes_S_M = self.handler.pbp_instance.network.params_S_M.get_value()
        bayes_S_V = self.handler.pbp_instance.network.params_S_V.get_value()

        np.savez('{}_{}'.format(filename, self.name), D=self.handler.D, K=self.handler.K, L=self.handler.L,
                 train_loss=self.train_loss, validation_loss=self.validation_loss,
                 train_f_meas=self.train_f_meas, validation_f_meas=self.validation_f_meas,
                 bayes_S_M=bayes_S_M, bayes_S_V=bayes_S_V, bayes_W_M=bayes_W_M, bayes_W_V=bayes_W_V)


class FrequentistListaRecorder(FrequentistRecorder):
    def __init__(self, handler, save_history, name):
        super().__init__(handler, save_history, name)
        if self.save_history:
            self.w_hist = []
            self.s_hist = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        super().train_and_record(beta_train, y_train, beta_validation, y_validation)

        if self.save_history:
            self.w_hist.append(self.handler.lista.W.get_value())
            self.s_hist.append(self.handler.lista.S.get_value())


class SequentialComparatorIstaFista(object):
    def __init__(self, D, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1,
                 train_freq=True, train_bayes=True, train_shared_bayes=True, use_ista=True, use_fista=True,
                 save_history=False):

        self.D = D
        self.K = K
        self.L = L

        self.data_generator = DataGenerator(D, K)

        self.train_freq = train_freq
        self.train_bayes = train_bayes
        self.train_shared_bayes = train_shared_bayes
        self.use_ista = use_ista
        self.use_fista = use_fista

        self.save_history = save_history

        self.recorders = []

        if self.train_freq:
            freq_lista = FrequentistListaHandler(D=D, K=K, L=L, X=self.data_generator.X, learning_rate=learning_rate)
            freq_lista_recorder = FrequentistListaRecorder(handler=freq_lista, save_history=save_history, name='freq lista')
            self.recorders.append(freq_lista_recorder)

        if self.train_bayes:
            separate_bayesian_lista = BayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            separate_bayesian_lista_recorder = SeparateBayesianRecorder(handler=separate_bayesian_lista, name='separate bayes lista')
            self.recorders.append(separate_bayesian_lista_recorder)

        if self.train_shared_bayes:
            shared_bayesian_lista = SingleBayesianListaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            shared_bayesian_lista_recorder = SharedBayesianRecorder(handler=shared_bayesian_lista, save_history=save_history, name='shared bayes lista')
            self.recorders.append(shared_bayesian_lista_recorder)

        if self.use_ista:
            ista = IstaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            ista_recorder = IstaRecorder(handler=ista, name='ista')
            self.recorders.append(ista_recorder)

        if self.use_fista:
            fista = FistaHandler(D=D, K=K, L=L, X=self.data_generator.X)
            fista_recorder = FistaRecorder(handler=fista, name='fista')
            self.recorders.append(fista_recorder)

        self.beta_train, self.y_train, _ = self.data_generator.new_sample(n_train_sample)
        self.beta_validation, self.y_validation, _ = self.data_generator.new_sample(n_validation_sample)

    def train_iteration(self):

        for recorder in self.recorders:
            recorder.train_and_record(beta_train=self.beta_train, y_train=self.y_train,
                                      beta_validation=self.beta_validation, y_validation=self.y_validation)

    def plot_quality_history(self):

        for recorder in self.recorders:
            plt.semilogy(recorder.train_loss, label="{}".format(recorder.name))
        plt.title('NMSE train')
        plt.legend()
        plt.show()

        for recorder in self.recorders:
            plt.semilogy(recorder.validation_loss, label="{}".format(recorder.name))
        plt.title('NMSE validation')
        plt.legend()
        plt.show()

        for recorder in self.recorders:
            plt.plot(recorder.train_f_meas, label="{}".format(recorder.name))
        plt.legend()
        plt.title('F measure train')
        plt.show()

        for recorder in self.recorders:
            plt.plot(recorder.validation_f_meas, label="{}".format(recorder.name))
        plt.legend()
        plt.title('F measure train')
        plt.show()

    def save_numpy(self, filename):
        for recorder in self.recorders:
            recorder.save_numpy(filename=filename)

        np.savez('{}_data'.format(filename), true_beta_train=self.beta_train,
                 true_beta_validation=self.beta_validation, y_train=self.y_train, y_validation=self.y_validation)


if __name__ == '__main__':

    rseed, D, K, L, batch_size, validation_size, n_iter = load_quick_experiment()
    np.random.seed(rseed)

    saved_comparator_file_name = []#'best_model_bayes_lista_single_matrices.pkl'

    if not saved_comparator_file_name:
        comparator = SequentialComparatorIstaFista(D, K, L, learning_rate=0.0001, n_train_sample=batch_size, n_validation_sample=validation_size)
    else:
        comparator = pickle.load(open(saved_comparator_file_name, 'rb'))

    for _ in range(n_iter):
        comparator.train_iteration()

    comparator.plot_quality_history()


    with open('cur.pkl', 'wb') as f:
        pickle.dump(comparator, f)

# D = 784, K = 100, L = 4, batch_size = 1000, validation_size = 100 - at the beginning bayes loss less than freg