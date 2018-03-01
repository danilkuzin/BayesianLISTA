from PBP_net_lista_single_matrices import pbp
from compare_freq_bayes.ListaHandler import ListaHandler
import numpy as np


class SingleBayesianListaHandler(ListaHandler):
    def __init__(self, D, K, L, X):
        super().__init__(D, K, L, X)

        mean_y_train = np.zeros(K)
        std_y_train = np.identity(K)

        self.pbp_instance = pbp.PBP_lista(L, D, K, mean_y_train, std_y_train)

    def train_iteration(self, beta_train, y_train):
        self.pbp_instance.do_pbp(beta_train, y_train, n_iterations=1)
        self.pbp_instance.sample_ws()
        return self.test(beta_train, y_train)

    def test(self, beta_test, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
        return self.mse(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
        return beta_estimator

    def mse(self, beta_true, beta_estimator):
        return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)))
