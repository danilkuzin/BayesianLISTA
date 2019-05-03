import matplotlib.pyplot as plt
import numpy as np

from .recorder import FrequentistListaRecorder, SharedBayesianRecorder, IstaRecorder, FistaRecorder
from ..algorithms.fista.handler import FistaHandler
from ..algorithms.ista.handler import IstaHandler
from ..algorithms.lista.handler import ListaHandler
from ..algorithms.listapbp.handler import SingleBayesianListaHandler


class SequentialComparator(object):
    def __init__(self, D, K, L, data, learning_rate, train_freq, train_shared_bayes, use_ista, use_fista,
                 save_history, initial_lambda):

        self.D = D
        self.K = K
        self.L = L

        # self.data_generator = DataGenerator(D, K)
        self.data = data

        self.train_freq = train_freq
        self.train_shared_bayes = train_shared_bayes
        self.use_ista = use_ista
        self.use_fista = use_fista

        self.save_history = save_history

        self.recorders = {}

        if self.train_freq:
            freq_lista = ListaHandler(D=D, K=K, L=L, X=self.data.X, learning_rate=learning_rate,
                                                 initial_lambda=initial_lambda)
            self.recorders['lista'] = FrequentistListaRecorder(handler=freq_lista, save_history=save_history,
                                                           name='freq lista')

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

    def train_iteration(self):
        for recorder_key in self.recorders.keys():
            self.recorders[recorder_key].train_and_record(beta_train=self.data.beta_train, y_train=self.data.y_train,
                                      beta_validation=self.data.beta_validation, y_validation=self.data.y_validation)

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
