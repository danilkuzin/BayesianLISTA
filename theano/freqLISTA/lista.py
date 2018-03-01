from PBP_net_lista.network_layer import theano_soft_threshold

import theano.tensor as T
import numpy as np
import theano


class Lista(object):

    def __init__(self, L, D, K, y, X):
        self.L = L
        self.D = D
        self.K = K

        W_init = X.T / (1.01 * np.linalg.norm(X,2)**2)
        self.W = theano.shared(value=W_init, name='W', borrow=True)
        S_init = np.identity(D) - np.matmul(W_init, X)
        self.S = theano.shared(value=S_init, name='S', borrow=True)
        self.params = [self.W, self.S]

        self.thr_lambda = theano.shared(value=0.1, name='thr_lambda', borrow=True)

        self.beta_estimator = self.net(y)
        self.y = y

    def mean_squared_error(self, beta):
        return T.mean(T.sqrt(T.sum(T.sqr(self.beta_estimator - beta), axis=1)))

    def normalised_mean_squared_error(self, beta):
        return T.mean(T.sqrt(T.sum(T.sqr(self.beta_estimator - beta), axis=1)) / T.sqrt(T.sum(T.sqr(beta), axis=1)))

    def net(self, y):
        b = T.dot(y, self.W.T)
        beta_estimator_history = [theano_soft_threshold(b, self.thr_lambda)]
        for l in range(1, self.L):
            c = b + T.dot(beta_estimator_history[-1], self.S.T)
            beta_estimator_history.append(theano_soft_threshold(c, self.thr_lambda))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

    def errors(self, beta):
        return self.mean_squared_error(beta)
