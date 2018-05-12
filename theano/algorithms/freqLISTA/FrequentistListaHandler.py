import theano.tensor as T
import theano
from tqdm import tqdm


from algorithms.freqLISTA.lista import Lista
import numpy as np

from algorithms.ListaHandler import ListaHandler


class FrequentistListaHandler(ListaHandler):
    def __init__(self, D, K, L, X, learning_rate, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)
        y = T.matrix('y')
        beta = T.matrix('beta')

        self.lista = Lista(L, D, K, y, X, initial_lambda)

        # cost = self.lista.mean_squared_error(beta)
        cost_nmse = self.lista.normalised_mean_squared_error(beta)
        cost_f_measure = self.lista.f_measure(beta)

        g_S = T.grad(cost=cost_nmse, wrt=self.lista.S)
        g_W = T.grad(cost=cost_nmse, wrt=self.lista.W)

        updates = [(self.lista.S, self.lista.S - learning_rate * g_S),
                   (self.lista.W, self.lista.W - learning_rate * g_W)]

        self.train_model = theano.function(
            inputs=[y, beta],
            outputs=cost_nmse,
            updates=updates
        )
        # self.test_model = theano.function(
        #     inputs=[y, beta],
        #     outputs=cost
        # )
        #
        self.predict_model = theano.function(
            inputs=[y],
            outputs=self.lista.beta_estimator
        )

        self.compute_nmse = theano.function(
            inputs=[y, beta],
            outputs=cost_nmse
        )

        self.compute_f_measure = theano.function(
            inputs=[y, beta],
            outputs=cost_f_measure
        )

    # def train_iteration(self, beta_train, y_train):
    #
    #     permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
    #                                    replace=False)
    #
    #     for i in permutation:
    #         self.train_model(np.expand_dims(y_train[i], axis=0), np.expand_dims(beta_train[i], axis=0))
    #
    #     mse = self.test_model(y=y_train, beta=beta_train)
    #     return mse

    def train_iteration(self, beta_train, y_train):

        permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
                                       replace=False)

        for i in permutation:
            self.train_model(np.expand_dims(y_train[i], axis=0), np.expand_dims(beta_train[i], axis=0))

        nmse = self.compute_nmse(y=y_train, beta=beta_train)
        f_measure = self.compute_f_measure(y=y_train, beta=beta_train)
        return nmse, f_measure

    # def train_iteration_batch(self, beta_train, y_train):
    #     mse = self.train_model(y_train, beta_train)
    #     return mse

    # def test(self, beta_test, y_test):
    #     mse = self.test_model(y=y_test, beta=beta_test)
    #     return mse

    def test(self, beta_test, y_test):
        nmse = self.compute_nmse(y=y_test, beta=beta_test)
        f_measure = self.compute_f_measure(y=y_test, beta=beta_test)
        return nmse, f_measure

    def predict(self, y_test):
        beta_estimator = self.predict_model(y=y_test)
        return beta_estimator
