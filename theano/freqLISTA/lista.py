from PBP_net_lista.network_layer import theano_soft_threshold

import theano.tensor as T
import numpy as np
import theano


class Lista(object):

    def __init__(self, L, D, K, y):
        self.L = L
        self.D = D
        self.K = K
        # self.batch_size = batch_size
        # self.dataGenerator = dataGenerator
        self.W = theano.shared(value=np.random.randn(D, K), name='W', borrow=True)
        self.S = theano.shared(value=np.random.randn(D, D), name='S', borrow=True)
        self.params = [self.W, self.S]

        self.thr_lambda = theano.shared(value=0.1, name='thr_lambda', borrow=True)
        #
        # self.y = T.vector()
        self.beta_estimator = self.net(y)
        self.y = y

    def mean_squared_error(self, beta):
        return T.mean(0.5 * T.sum(T.sqr(self.beta_estimator - beta)))

    # def train(self, Beta_train, Y_train):
    #     beta_estimator = self.net(Y_train_batch)
    #
    #
    #     # inputs = [input_scalar]
    #     # outputs = [scalar_times_shared]
    #     #
    #     # my_updates = {
    #     #     shared_vector_1: scalar_times_shared  # и этот же результат запишем в shared_vector_1
    #     # }
    #     #
    #     # compute_and_save = theano.function(inputs, outputs, updates=my_updates)
    #
    #     loss = self.loss(self.net(y_batch), beta_batch)
    #     d_loss_wrt_params = T.grad(beta_estimator, [self.S, self.W])
    #     d_S = d_loss_wrt_params[0]
    #     d_W = d_loss_wrt_params[1]
    #     self.S -= self.learning_rate * d_S
    #     self.W -= self.learning_rate * d_W
    #
    #     updates = [(self.S, self.S - self.learning_rate * d_S),
    #                (self.W, self.W - self.learning_rate * d_W)]
    #     MSGD = theano.function([y_batch, beta_batch], loss, updates=updates)
    #
    #     while True:
    #         beta_batch, y_batch, _ = self.dataGenerator.new_sample(self.batch_size)
    #
    #         print('Current loss is ', MSGD(y_batch, beta_batch))
    #         if < stopping condition is met >:
    #             return params
    #
    # def test(self, Y_test):
    #     pass

    def net(self, y):
        B = T.dot(self.W, y)
        beta_estimator_history = [theano_soft_threshold(B, self.thr_lambda)]
        for l in range(1, self.L):
            C = B + T.dot(self.S, beta_estimator_history[-1])
            beta_estimator_history.append(theano_soft_threshold(C, self.thr_lambda))
        beta_estimator = beta_estimator_history[-1]
        return beta_estimator

    def errors(self, beta):
        return self.mean_squared_error(beta)
