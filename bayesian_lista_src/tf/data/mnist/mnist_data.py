import numpy as np
import tensorflow as tf


class MnistData(object):
    def __init__(self, K, train_size, valid_size):
        self.beta_train = None
        self.train_labels = None
        self.beta_validation = None
        self.validation_labels = None

        self.K = K
        self.X = None
        self.image_size = (28, 28)

        self.y_train = None
        self.y_validation = None

        self.training_size = train_size
        self.validation_size = valid_size

    def check_download(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        random_train_ind = np.random.choice(mnist.train.images.shape[0], self.training_size, replace=False)
        self.beta_train = mnist.train.images[random_train_ind].astype(np.float32)
        random_validation_ind = np.random.choice(mnist.validation.images.shape[0], self.validation_size, replace=False)
        self.beta_validation = mnist.validation.images[random_validation_ind].astype(np.float32)

    def random_dictionary(self, normalise):
        self.X = np.random.randn(self.K, self.beta_train.shape[1]).astype(np.float32)
        if normalise:
            self.X = np.dot(self.X, np.diag(1 / np.linalg.norm(self.X, axis=0)))

        self.y_train = np.dot(self.beta_train, self.X.T).astype(np.float32)
        self.y_validation = np.dot(self.beta_validation, self.X.T).astype(np.float32)

