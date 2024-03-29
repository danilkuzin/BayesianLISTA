from time import time

import tensorflow as tf
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
import matplotlib.pyplot as plt

class MnistData(object):
    def __init__(self, K, train_size, valid_size):
        self.beta_train = None
        self.train_labels = None
        self.dictionary_learn_data = None
        self.dictionary_learn_labels = None
        self.beta_validation = None
        self.validation_labels = None

        self.K = K
        self.X = None
        self.image_size = (28, 28)

        self.y_train = None
        self.y_validation = None

        self.training_size = train_size
        self.validation_size = valid_size

    def check_download(self, normalise):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        random_train_ind = np.random.choice(mnist.train.images.shape[0], self.training_size, replace=False)
        self.beta_train = mnist.train.images[random_train_ind]
        #self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.dictionary_learn_data = mnist.test.images
        #self.dictionary_learn_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        random_validation_ind = np.random.choice(mnist.validation.images.shape[0], self.validation_size, replace=False)
        self.beta_validation = mnist.validation.images[random_validation_ind]
        #self.validation_labels = np.asarray(mnist.validation.labels, dtype=np.int32)

        if normalise:
            self.normalize()

    def normalize(self):
        self.train_mean = np.mean(np.append(self.beta_train, self.beta_validation, axis=0), axis=0)
        self.train_std = np.std(np.append(self.beta_train, self.beta_validation, axis=0), axis=0)
        self.beta_train = self.beta_train - self.train_mean
        self.beta_train[:, self.train_std != 0] = self.beta_train[:, self.train_std != 0] / \
                                                  self.train_std[self.train_std != 0]
        self.beta_validation = self.beta_validation - self.train_mean
        self.beta_validation[:, self.train_std != 0] = self.beta_validation[:, self.train_std != 0] / \
                                                       self.train_std[self.train_std != 0]

    def learn_dictionary(self):
        print('Learning the dictionary...')
        t0 = time()
        dico = MiniBatchDictionaryLearning(n_components=self.K, alpha=1, n_iter=500)
        self.X = dico.fit(self.dictionary_learn_data).components_
        dt = time() - t0
        print('done in %.2fs.' % dt)

        self.y_train = np.dot(self.beta_train, self.X.T)
        self.y_validation = np.dot(self.beta_validation, self.X.T)

    def random_dictionary(self, normalise):
        self.X = np.random.randn(self.K, self.beta_train.shape[1])
        if normalise:
            self.X = np.dot(self.X, np.diag(1 / np.linalg.norm(self.X, axis=0)))

        self.y_train = np.dot(self.beta_train, self.X.T)
        self.y_validation = np.dot(self.beta_validation, self.X.T)


    def generate_random_design_matrix(self):
        self.X = np.random.randn(self.K, self.beta_train.shape[1])

        self.y_train = np.dot(self.beta_train, self.X.T)
        self.y_validation = np.dot(self.beta_validation, self.X.T)

    def plot_learnt_dictionary(self):
        for i, comp in enumerate(self.X[:10]):
            plt.imshow(comp.reshape(self.image_size), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.show()


if __name__=='__main__':
    data=MnistData()
    data.check_download()
    data.learn_dictionary()
    data.plot_learnt_dictionary()
