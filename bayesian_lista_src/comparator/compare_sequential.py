import numpy as np
import six.moves.cPickle as pickle

from algorithms.PBP_net_lista.BayesianListaHandler import BayesianListaHandler
from algorithms.PBP_net_lista.DataGenerator import DataGenerator
from algorithms.PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler

from comparator.recorder import FrequentistListaRecorder, SeparateBayesianRecorder, SharedBayesianRecorder, \
    IstaRecorder, FistaRecorder
from experiments.synthetic.experiments_parameters import load_synthetic_experiment_2, load_quick_experiment
from algorithms.fista.fIstaHandler import FistaHandler
from algorithms.freqLISTA.FrequentistListaHandler import FrequentistListaHandler

import matplotlib.pyplot as plt

from algorithms.ista.IstaHandler import IstaHandler


class SequentialComparator(object):
    def __init__(self, D, K, L, data, learning_rate, train_freq, train_bayes, train_shared_bayes, use_ista, use_fista,
                 save_history, initial_lambda):

        self.D = D
        self.K = K
        self.L = L

        # self.data_generator = DataGenerator(D, K)
        self.data = data

        self.train_freq = train_freq
        self.train_bayes = train_bayes
        self.train_shared_bayes = train_shared_bayes
        self.use_ista = use_ista
        self.use_fista = use_fista

        self.save_history = save_history

        self.recorders = {}

        if self.train_freq:
            freq_lista = FrequentistListaHandler(D=D, K=K, L=L, X=self.data.X, learning_rate=learning_rate,
                                                 initial_lambda=initial_lambda)
            self.recorders['lista'] = FrequentistListaRecorder(handler=freq_lista, save_history=save_history,
                                                           name='freq lista')

        if self.train_bayes:
            separate_bayesian_lista = BayesianListaHandler(D=D, K=K, L=L, X=self.data.X,
                                                           initial_lambda=initial_lambda)
            self.recorders['separate_bayes'] = SeparateBayesianRecorder(handler=separate_bayesian_lista,
                                                                        name='separate bayes lista')

        if self.train_shared_bayes:
            shared_bayesian_lista = SingleBayesianListaHandler(D=D, K=K, L=L, X=self.data.X,
                                                               initial_lambda=initial_lambda)
            self.recorders['shared_bayes'] = SharedBayesianRecorder(handler=shared_bayesian_lista,
                                                                    save_history=save_history,
                                                                    name='shared bayes lista')

        if self.use_ista:
            ista = IstaHandler(D=D, K=K, L=L, X=self.data.X, initial_lambda=initial_lambda)
            self.recorders['ista'] = IstaRecorder(handler=ista, name='ista')

        if self.use_fista:
            fista = FistaHandler(D=D, K=K, L=L, X=self.data.X, initial_lambda=initial_lambda)
            self.recorders['fista'] = FistaRecorder(handler=fista, name='fista')

        # self.beta_train, self.y_train, _ = self.data_generator.new_sample(n_train_sample)
        # self.beta_validation, self.y_validation, _ = self.data_generator.new_sample(n_validation_sample)

    def train_iteration(self):
        for recorder_key in self.recorders.keys():
            self.recorders[recorder_key].train_and_record(beta_train=self.data.beta_train, y_train=self.data.y_train,
                                      beta_validation=self.data.beta_validation, y_validation=self.data.y_validation)

    # def get_final_statistics(self):
    #     res = {}
    #     for recorder in self.recorders:
    #         res[recorder.name].train_loss = recorder.train_loss[-1]
    #         res[recorder.name].validation_loss = recorder.validation_loss[-1]
    #         res[recorder.name].train_f_meas = recorder.train_f_meas[-1]
    #         res[recorder.name].validation_f_meas = recorder.validation_f_meas[-1]
    #         res[recorder.name].time = recorder.time[-1]
    #
    #     return res

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

        np.savez('{}_data'.format(filename), true_beta_train=self.data.beta_train,
                 true_beta_validation=self.data.beta_validation, y_train=self.data.y_train, y_validation=self.data.y_validation)


# if __name__ == '__main__':
#
#     rseed, D, K, L, batch_size, validation_size, n_iter = load_quick_experiment()
#     np.random.seed(rseed)
#
#     saved_comparator_file_name = []  # 'best_model_bayes_lista_single_matrices.pkl'
#
#     if not saved_comparator_file_name:
#         comparator = SequentialComparatorIstaFista(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
#                                                    n_validation_sample=validation_size)
#     else:
#         comparator = pickle.load(open(saved_comparator_file_name, 'rb'))
#
#     for _ in range(n_iter):
#         comparator.train_iteration()
#
#     comparator.plot_quality_history()
#
#     with open('cur.pkl', 'wb') as f:
#         pickle.dump(comparator, f)

# D = 784, K = 100, L = 4, batch_size = 1000, validation_size = 100 - at the beginning bayes loss less than freg
