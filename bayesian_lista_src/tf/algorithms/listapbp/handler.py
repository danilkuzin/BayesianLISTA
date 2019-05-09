from ..listapbp import pbp
from ..handler import Handler
import numpy as np


class SingleBayesianListaHandler(Handler):
    def __init__(self, D, K, L, X, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        mean_y_train = np.zeros(K)
        std_y_train = np.identity(K)

        self.pbp_instance = pbp.PBP_lista(L, D, K, X, mean_y_train, std_y_train, initial_lambda)

    def train(self, num_epochs, beta_train, y_train):
        Ws, Ss, thr_lambdas = [], [], []
        epochs = range(num_epochs)
        for epoch in epochs:
             self.train_iteration(beta_train, y_train)

    def train_iteration(self, beta_train, y_train, sample_mean=True):
        self.pbp_instance.train(beta_train, y_train, n_iterations=1)
        if sample_mean:
            self.pbp_instance.sample_mean_ws()
        else:
            self.pbp_instance.sample_ws()
        return self.test(beta_train, y_train)

    def test(self, beta_test, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

    def predict(self, y_test):
        beta_estimator = self.pbp_instance.get_deterministic_output(y_test)
        return beta_estimator

    def predict_probabilistic(self, y_test):
        w_list = []
        m_list = []
        v_list = []
        for i in range(y_test.shape[0]):
            w, m, v = self.pbp_instance.predict_probabilistic(y_test[i])
            w_list.append(w)
            m_list.append(m)
            v_list.append(v)
        return np.array(w_list), np.array(m_list), np.array(v_list)

    def nmse(self, beta_true, beta_estimator):
        return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)) / np.sqrt(np.sum(beta_true**2, axis=1)))

