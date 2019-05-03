from tf.algorithms.handler import Handler
import numpy as np

from tf.algorithms.fista.fista import Fista


class FistaHandler(Handler):
    def __init__(self, D, K, L, X, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        self.model = Fista(L, D, K, X, initial_lambda)

    def test(self, beta_test, y_test):
        beta_estimator = self.predict(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.model(y_test)
        return beta_estimator

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



