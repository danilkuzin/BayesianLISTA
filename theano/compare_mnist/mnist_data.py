from time import time

import tensorflow as tf
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
import matplotlib.pyplot as plt

class MnistData(object):
    def __init__(self, K=100):
        self.train_data = None
        self.train_labels = None
        self.dictionary_learn_data = None
        self.dictionary_learn_labels = None
        self.validation_data = None
        self.validation_labels = None

        self.K = K
        self.X = None
        self.image_size = (28, 28)

        self.y_train = None
        self.y_validation = None

        self.training_size = 5000

    def check_download(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        random_train_ind = np.random.choice(mnist.train.images.shape[0], self.training_size, replace=False)
        self.train_data = mnist.train.images[random_train_ind]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.dictionary_learn_data = mnist.test.images
        self.dictionary_learn_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        self.validation_data = mnist.validation.images
        self.validation_labels = np.asarray(mnist.validation.labels, dtype=np.int32)

    def learn_dictionary(self):
        print('Learning the dictionary...')
        t0 = time()
        dico = MiniBatchDictionaryLearning(n_components=self.K, alpha=1, n_iter=500)
        self.X = dico.fit(self.dictionary_learn_data).components_
        dt = time() - t0
        print('done in %.2fs.' % dt)

        self.y_train = np.dot(self.train_data, self.X.T)
        self.y_validation = np.dot(self.validation_data, self.X.T)


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
