import tensorflow as tf
from tqdm import trange

from tf.algorithms.handler import Handler
from tf.algorithms.lista.lista import Lista

import time


class ListaHandler(Handler):
    def __init__(self, D, K, L, X, learning_rate, initial_lambda):
        super().__init__(D, K, L, X, initial_lambda)

        self.model = Lista(L, D, K, X, initial_lambda)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        self.train_loss = tf.contrib.eager.metrics.Mean()
        self.validation_loss = tf.contrib.eager.metrics.Mean()
        self.train_f_measure = tf.contrib.eager.metrics.Mean()

    def train_iteration(self, beta_train, y_train):
        with tf.GradientTape() as t:
            predictions = self.model(y_train)
            current_loss = tf.losses.mean_squared_error(labels=beta_train, predictions=predictions)
        grads = t.gradient(current_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss(current_loss)
        self.train_f_measure(tf.contrib.metrics.f1_score(labels=beta_train, predictions=predictions))

        #return current_loss.numpy(), Handler.f_measure(beta_train, self.model(y_train))

    def train(self, n_epochs, train_data):
        train_summary_writer = tf.contrib.summary.create_file_writer("logs")

        t = trange(n_epochs, desc='ML')
        for epoch in t:
            start_time = time.process_time()
            for i, (beta_batch, y_batch) in enumerate(train_data):
                self.train_iteration(beta_train=beta_batch, y_train=y_batch)
            elapsed_time = time.process_time() - start_time

            # train_pred = self.model(y_train.astype(np.float32))
            # valid_pred = self.model(y_valid.astype(np.float32))
            # loss = ListaHandler.loss(train_pred, beta_train)
            # f_meas = ListaHandler.f_measure(beta_train, train_pred)
            # valid_loss = ListaHandler.loss(valid_pred, beta_valid)
            # valid_f_meas = ListaHandler.f_measure(beta_valid, valid_pred)
            # t.set_description(
            #     f'ML (loss={loss.numpy():.3f}, valid_loss={valid_loss.numpy():.3f} f-meas:{f_meas:.3f} valid_f-meas:{valid_f_meas:.3f})')
            # loss_hist.append(loss.numpy())
            # valid_loss_hist.append(valid_loss.numpy())
            # f_meas_hist.append(f_meas)
            # valid_f_meas_hist.append(valid_f_meas)
            # times.append(elapsed_time)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result())

            template = 'Epoch {}, Loss: {}'
            print(template.format(epoch + 1,
                              self.train_loss.result()))

    def predict(self, y_test):
        beta_estimator = self.model(y_test)
        return beta_estimator

    def test(self, beta_test, y_test):
        beta_estimator = self.predict(y_test)
        return self.nmse(beta_test, beta_estimator), self.f_measure(beta_test, beta_estimator)

