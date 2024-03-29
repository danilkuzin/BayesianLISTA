import theano.tensor as T
import theano
from tqdm import tqdm

from compare_freq_bayes.ListaHandler import ListaHandler
from freqLISTA.lista import Lista
import numpy as np

class FrequentistListaHandler(ListaHandler):
    def __init__(self, D, K, L, X, learning_rate):
        super().__init__(D, K, L, X)
        y = T.matrix('y')
        beta = T.matrix('beta')
        learning_rate = T.scalar('lr')

        lista = Lista(L, D, K, y, X)

        cost = lista.mean_squared_error(beta)

        cost_nmse = lista.normalised_mean_squared_error(beta)

        g_S = T.grad(cost=cost, wrt=lista.S)
        g_W = T.grad(cost=cost, wrt=lista.W)
        g_lambda = T.grad(cost=cost, wrt=lista.thr_lambda)


        updates = [(lista.S, lista.S - learning_rate * g_S),
                   (lista.W, lista.W - learning_rate * g_W),
                   (lista.thr_lambda, lista.thr_lambda - learning_rate * g_lambda)]

        self.train_model = theano.function(
            inputs=[y, beta, learning_rate],
            outputs=cost,
            updates=updates
        )

        self.test_model = theano.function(
            inputs=[y, beta],
            outputs=cost
        )

        self.predict_model = theano.function(
            inputs=[y],
            outputs=lista.beta_estimator
        )

        self.compute_nmse = theano.function(
            inputs=[y, beta],
            outputs=cost_nmse
        )

        self.check_lambda = theano.function(
            inputs=[],
            outputs=lista.thr_lambda
        )

    def train_iteration(self, beta_train, y_train):

        permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
                                       replace=False)

        for i in permutation:
            self.train_model(np.expand_dims(y_train[i], axis=0), np.expand_dims(beta_train[i], axis=0))

        mse = self.test_model(y=y_train, beta=beta_train)
        return mse

    def train_iteration_nmse(self, beta_train, y_train, learning_rate):

        permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
                                       replace=False)

        self.train_model(y_train[permutation[:1000],:], beta_train[permutation[:1000],:], learning_rate)
        print(self.check_lambda())
        nmse = self.compute_nmse(y=y_train, beta=beta_train)
        return nmse

    def train_iteration_nmse_no_batches(self, beta_train, y_train):

        permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
                                       replace=False)

        for i in permutation:
            self.train_model(np.expand_dims(y_train[i], axis=0), np.expand_dims(beta_train[i], axis=0))
        nmse = self.compute_nmse(y=y_train, beta=beta_train)
        return nmse


    def train_iteration_batch(self, beta_train, y_train):
        mse = self.train_model(y_train, beta_train)
        return mse

    def test(self, beta_test, y_test):
        mse = self.test_model(y=y_test, beta=beta_test)
        return mse

    def test_nmse(self, beta_test, y_test):
        nmse = self.compute_nmse(y=y_test, beta=beta_test)
        return nmse

    def predict(self, y_test):
        beta_estimator = self.predict_model(y=y_test)
        return beta_estimator
