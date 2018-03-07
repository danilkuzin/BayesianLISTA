from PBP_net_lista import pbp
from compare_freq_bayes.ListaHandler import ListaHandler
import numpy as np

from fista.fista import Fista
from ista.ista import Ista


class FistaHandler(ListaHandler):
    def __init__(self, D, K, L, X):
        super().__init__(D, K, L, X)

        self.fista = Fista(L, D, K, X)

    def test(self, beta_test, y_test):
        beta_estimator = self.predict(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.fista.predict(y_test)
        return beta_estimator

    def nmse(self, beta_true, beta_estimator):
        return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)) / np.sqrt(np.sum(beta_true**2, axis=1)))

    def f_measure(self, beta_true, beta_estimator):
        true_zero_loc = beta_true == 0
        true_nonzero_loc = beta_true != 0
        est_zero_loc = beta_estimator == 0
        est_nonzero_loc = beta_estimator != 0

        tp = np.sum(true_nonzero_loc * est_nonzero_loc)
        fp = np.sum(true_zero_loc * est_nonzero_loc)
        fn = np.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall > 0:
            return 2 * (precision * recall / (precision + recall))
        else:
            return 0

    def train_iteration(self, beta_train, y_train):
        return self.test(beta_train, y_train)



