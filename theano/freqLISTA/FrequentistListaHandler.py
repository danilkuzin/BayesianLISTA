import theano.tensor as T
import theano

from compare_freq_bayes.ListaHandler import ListaHandler
from freqLISTA.lista import Lista


class FrequentistListaHandler(ListaHandler):
    def __init__(self, D, K, L, X, learning_rate):
        super().__init__(D, K, L, X)
        y = T.matrix('y')
        beta = T.matrix('beta')

        lista = Lista(L, D, K, y, X)

        cost = lista.mean_squared_error(beta)

        g_S = T.grad(cost=cost, wrt=lista.S)
        g_W = T.grad(cost=cost, wrt=lista.W)

        updates = [(lista.S, lista.S - learning_rate * g_S),
                   (lista.W, lista.W - learning_rate * g_W)]

        self.train_model = theano.function(
            inputs=[y, beta],
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

    def train_iteration(self, beta_train, y_train):
        mse = self.train_model(y_train, beta_train)
        return mse

    def test(self, beta_test, y_test):
        mse = self.test_model(y=y_test, beta=beta_test)
        return mse

    def predict(self, y_test):
        beta_estimator = self.predict_model(y=y_test)
        return beta_estimator
