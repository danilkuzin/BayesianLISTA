from tf.algorithms.shared.soft_thresholding import soft_threshold

import tensorflow as tf


class Lista:

    def __init__(self, L, D, K, X, initial_lambda):
        self.L = L
        self.D = D
        self.K = K
        self.initial_lambda = initial_lambda

        W_init = tf.transpose(X) / (1.01 * tf.norm(X,2)**2)
        self.W = tf.Variable(initial_value=W_init, name='W')
        S_init = tf.eye(D, D) - tf.matmul(W_init, X)
        self.S = tf.Variable(initial_value=S_init, name='S')
        #self.params = [self.W, self.S]

        self.thr_lambda = tf.Variable(initial_value=self.initial_lambda, name='thr_lambda')

        #self.beta_estimator = self.net(y)
        #self.y = y

    # def normalised_mean_squared_error(self, beta):
    #     return T.mean(T.sqrt(T.sum(T.sqr(self.beta_estimator - beta), axis=1)) / T.sqrt(T.sum(T.sqr(beta), axis=1)))

    # def f_measure(self, beta):
    #     true_zero_loc = T.eq(beta, 0)
    #     true_nonzero_loc = T.neq(beta, 0)
    #     est_zero_loc = T.eq(self.beta_estimator, 0)
    #     est_nonzero_loc = T.neq(self.beta_estimator, 0)
    #
    #     tp = T.sum(true_nonzero_loc * est_nonzero_loc)
    #     fp = T.sum(true_zero_loc * est_nonzero_loc)
    #     fn = T.sum(true_nonzero_loc * est_zero_loc)
    #
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #
    #     f_meas = T.switch(T.gt(precision + recall, 0), 2 * (precision * recall / (precision + recall)), 0)
    #
    #     return f_meas

    def __call__(self, y):
        b = tf.matmul(y, tf.transpose(self.W))
        beta_estimator_history = [soft_threshold(b, self.thr_lambda)]
        for l in range(1, self.L):
            c = b + tf.matmul(beta_estimator_history[-1], tf.transpose(self.S))
            beta_estimator_history.append(soft_threshold(c, self.thr_lambda))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

