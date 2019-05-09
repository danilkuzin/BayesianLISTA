import tensorflow as tf
import numpy as np

class Handler(object):
    def __init__(self, D, K, L, X, initial_lambda):
        self.D = D
        self.K = K
        self.L = L
        self.X = X
        self.initial_lambda = initial_lambda

    @staticmethod
    def loss(predicted_beta, desired_beta):
        return tf.reduce_mean(tf.square(predicted_beta - desired_beta))

    def train_iteration(self, beta_train, y_train):
        raise NotImplementedError

    def test(self, beta_test, y_test):
        raise NotImplementedError

    def predict(self, y_test):
        raise NotImplementedError

    @staticmethod
    def f_measure(beta_true, beta_estimator):
        true_zero_loc = tf.equal(beta_true, 0)
        true_nonzero_loc = tf.not_equal(beta_true, 0)
        est_zero_loc = tf.equal(beta_estimator, 0)
        est_nonzero_loc = tf.not_equal(beta_estimator, 0)

        tp = tf.count_nonzero(tf.equal(true_nonzero_loc, est_nonzero_loc))
        fp = tf.count_nonzero(tf.equal(true_zero_loc, est_nonzero_loc))
        fn = tf.count_nonzero(tf.equal(true_nonzero_loc, est_zero_loc))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall > 0:
            return 2 * (precision * recall / (precision + recall))
        else:
            return 0

    @staticmethod
    def nmse(beta_true, beta_estimator):
        return np.mean(np.sqrt(np.sum((beta_estimator - beta_true)**2, axis=1)) / np.sqrt(np.sum(beta_true**2, axis=1)))
