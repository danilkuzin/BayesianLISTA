from ..handler import Handler
import numpy as np

from ..ista.ista import Ista


class IstaHandler(Handler):
    def __init__(self, D, K, L, X, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        self.model = Ista(L, D, K, X, initial_lambda)

    def train_iteration(self, beta_train, y_train):
        return self.test(beta_train, y_train)

    def test(self, beta_test, y_test):
        beta_estimator = self.predict(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.model(y_test)
        return beta_estimator

    def nmse(self, beta_true, beta_estimator):
        return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)) / np.sqrt(np.sum(beta_true**2, axis=1)))





