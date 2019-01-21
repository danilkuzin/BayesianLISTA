import tensorflow as tf


from tf.algorithms.lista.lista import Lista

from tf.algorithms.handler import Handler


class ListaHandler(Handler):
    def __init__(self, D, K, L, X, learning_rate, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)
        # y = T.matrix('y')
        # beta = T.matrix('beta')
        #
        self.model = Lista(L, D, K, X, initial_lambda)
        self.learning_rate = learning_rate

        #cost_nmse = self.lista.normalised_mean_squared_error(beta)
        #cost_f_measure = self.lista.f_measure(beta)

        # g_S = T.grad(cost=cost_nmse, wrt=self.lista.S)
        # g_W = T.grad(cost=cost_nmse, wrt=self.lista.W)
        #
        # updates = [(self.lista.S, self.lista.S - learning_rate * g_S),
        #            (self.lista.W, self.lista.W - learning_rate * g_W)]
        #
        # self.train_model = theano.function(
        #     inputs=[y, beta],
        #     outputs=cost_nmse,
        #     updates=updates
        # )
        # self.test_model = theano.function(
        #     inputs=[y, beta],
        #     outputs=cost
        # )
        #
        # self.predict_model = theano.function(
        #     inputs=[y],
        #     outputs=self.lista.beta_estimator
        # )
        #
        # self.compute_nmse = theano.function(
        #     inputs=[y, beta],
        #     outputs=cost_nmse
        # )
        #
        # self.compute_f_measure = theano.function(
        #     inputs=[y, beta],
        #     outputs=cost_f_measure
        # )

    # use non-normalised MSE here
    @staticmethod
    def loss(predicted_beta, desired_beta):
        return tf.reduce_mean(tf.square(predicted_beta - desired_beta))

    def train_iteration(self, beta_train, y_train):
        with tf.GradientTape() as t:
            current_loss = ListaHandler.loss(self.model(y_train), beta_train)
        dW, dS, dLambda = t.gradient(current_loss, [self.model.W, self.model.S, self.model.thr_lambda])
        self.model.W.assign_sub(self.learning_rate * dW)
        self.model.S.assign_sub(self.learning_rate * dS)
        self.model.thr_lambda.assign_sub(self.learning_rate * dLambda)

    def train(self, num_epochs, beta_train, y_train):
        Ws, Ss, thr_lambdas = [], [], []
        epochs = range(num_epochs)
        for epoch in epochs:
        #     Ws.append(self.model.W.numpy())
        #     Ss.append(self.model.S.numpy())
        #     thr_lambdas.append(self.model.thr_lambda.numpy())
        #     current_loss = ListaHandler.loss(self.model(y_train), beta_train)
        #
             self.train_iteration(beta_train, y_train)
        #     print('Epoch %2d: W=%1.2f S=%1.2f lambda=%1.2f, loss=%2.5f' %
        #           (epoch, Ws[-1], Ss[-1], thr_lambdas[-1], current_loss))

        # Let's plot it all
        # plt.plot(epochs, Ws, 'r',
        #          epochs, bs, 'b')
        # plt.plot([TRUE_W] * len(epochs), 'r--',
        #          [TRUE_b] * len(epochs), 'b--')
        # plt.legend(['W', 'b', 'true W', 'true_b'])
        # plt.show()

# def train_iteration(self, beta_train, y_train):
    #
    #     permutation = np.random.choice(range(beta_train.shape[0]), beta_train.shape[0],
    #                                    replace=False)
    #
    #     for i in permutation:
    #         self.train_model(np.expand_dims(y_train[i], axis=0), np.expand_dims(beta_train[i], axis=0))
    #
    #     nmse = self.compute_nmse(y=y_train, beta=beta_train)
    #     f_measure = self.compute_f_measure(y=y_train, beta=beta_train)
    #     return nmse, f_measure
    #
    # def test(self, beta_test, y_test):
    #     nmse = self.compute_nmse(y=y_test, beta=beta_test)
    #     f_measure = self.compute_f_measure(y=y_test, beta=beta_test)
    #     return nmse, f_measure
    #
    def predict(self, y_test):
        beta_estimator = self.model(y_test)
        return beta_estimator
