
import numpy as np

from algorithms.ListaHandler import ListaHandler
from algorithms.PBP_net_lista import pbp


class BayesianListaHandler(ListaHandler):
    def __init__(self, D, K, L, X, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        mean_y_train = np.zeros(K)
        std_y_train = np.identity(K)

        self.pbp_instance = pbp.PBP_lista(L, D, K, X, mean_y_train, std_y_train, initial_lambda)

    # def train_iteration(self, beta_train, y_train, sample_mean=False):
    #     self.pbp_instance.do_pbp(beta_train, y_train, n_iterations=1)
    #     if sample_mean:
    #         self.pbp_instance.sample_mean_ws()
    #     else:
    #         self.pbp_instance.sample_ws()
    #     return self.test(beta_train, y_train)

    def train_iteration(self, beta_train, y_train, sample_mean=True):
        self.pbp_instance.do_pbp(beta_train, y_train, n_iterations=1)
        if sample_mean:
            self.pbp_instance.sample_mean_ws()
        else:
            self.pbp_instance.sample_ws()
        return self.test(beta_train, y_train)

    # def test(self, beta_test, y_test):
    #     beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
    #     return self.mse(beta_test, beta_estimator)

    def test(self, beta_test, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)

        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
        return beta_estimator

    # def mse(self, beta_true, beta_estimator):
    #     return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)))

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



