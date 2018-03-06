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

    # def mean_squared_error(self, beta):
    #     return T.mean(T.sqrt(T.sum(T.sqr(self.beta_estimator - beta), axis=1)))

    def normalised_mean_squared_error(self, beta):
        return T.mean(T.sqrt(T.sum(T.sqr(self.beta_estimator - beta), axis=1)) / T.sqrt(T.sum(T.sqr(beta), axis=1)))

    def f_measure(self, beta):
        true_zero_loc = T.eq(beta, 0)
        true_nonzero_loc = T.neq(beta, 0)
        est_zero_loc = T.eq(self.beta_estimator, 0)
        est_nonzero_loc = T.neq(self.beta_estimator, 0)

        tp = T.sum(true_nonzero_loc * est_nonzero_loc)
        fp = T.sum(true_zero_loc * est_nonzero_loc)
        fn = T.sum(true_nonzero_loc * est_zero_loc)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f_meas = T.switch(T.gt(precision + recall, 0), 2 * (precision * recall / (precision + recall)), 0)

        return f_meas

    def net(self, y):
        b = T.dot(y, self.W.T)
        beta_estimator_history = [theano_soft_threshold(b, self.thr_lambda)]
        for l in range(1, self.L):
            c = b + T.dot(beta_estimator_history[-1], self.S.T)
            beta_estimator_history.append(theano_soft_threshold(c, self.thr_lambda))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

    # def errors(self, beta):
    #     return self.mean_squared_error(beta)
