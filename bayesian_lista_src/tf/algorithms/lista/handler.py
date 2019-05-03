import tensorflow as tf

from tf.algorithms.handler import Handler
from tf.algorithms.lista.lista import Lista


class ListaHandler(Handler):
    def __init__(self, D, K, L, X, learning_rate, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        self.model = Lista(L, D, K, X, initial_lambda)
        self.learning_rate = learning_rate

    def train_iteration(self, beta_train, y_train):
        with tf.GradientTape() as t:
            current_loss = ListaHandler.loss(self.model(y_train), beta_train)
        dW, dS, dLambda = t.gradient(current_loss, [self.model.W, self.model.S, self.model.thr_lambda])
        self.model.W.assign_sub(self.learning_rate * dW)
        self.model.S.assign_sub(self.learning_rate * dS)
        self.model.thr_lambda.assign_sub(self.learning_rate * dLambda)
        return current_loss.numpy(), Handler.f_measure(beta_train, self.model(y_train))

    def train(self, num_epochs, beta_train, y_train):
        Ws, Ss, thr_lambdas = [], [], []
        epochs = range(num_epochs)
        for epoch in epochs:
             self.train_iteration(beta_train, y_train)

    def predict(self, y_test):
        beta_estimator = self.model(y_test)
        return beta_estimator

    def test(self, beta_test, y_test):
        beta_estimator = self.predict(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

