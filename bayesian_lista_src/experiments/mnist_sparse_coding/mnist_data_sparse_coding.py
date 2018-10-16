from time import time

import tensorflow as tf
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
import matplotlib.pyplot as plt

from skimage.transform import resize


class MnistDataSparseCoding(object):
    def __init__(self, D, train_size, valid_size, image_size):
        self.beta_train = None
        self.train_labels = None
        self.dictionary_learn_images = None
        self.dictionary_learn_labels = None
        self.beta_validation = None
        self.validation_labels = None

        self.D = D
        self.X = None
        self.image_size = (image_size, image_size)

        self.y_train = None
        self.y_validation = None

        self.training_size = train_size
        self.validation_size = valid_size

    def check_download(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")

        train_images = np.array([resize(np.reshape(q, (28, 28)), self.image_size) for q in mnist.train.images])
        train_images = np.reshape(train_images, (train_images.shape[0], self.image_size[0]**2))
        validation_images = np.array([resize(np.reshape(q, (28, 28)), self.image_size) for q in mnist.validation.images])
        validation_images = np.reshape(validation_images, (validation_images.shape[0], self.image_size[0]**2))
        test_images = np.array([resize(np.reshape(q, (28, 28)), self.image_size) for q in mnist.test.images])
        test_images = np.reshape(test_images, (test_images.shape[0], self.image_size[0]**2))

        random_train_ind = np.random.choice(train_images.shape[0], self.training_size, replace=False)
        self.y_train = train_images[random_train_ind]
        #self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.dictionary_learn_images = test_images
        #self.dictionary_learn_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        random_validation_ind = np.random.choice(validation_images.shape[0], self.validation_size, replace=False)
        self.y_validation = validation_images[random_validation_ind]
        #self.validation_labels = np.asarray(mnist.validation.labels, dtype=np.int32)


    def learn_dictionary(self):
        print('Learning the dictionary...')
        t0 = time()
        dico = MiniBatchDictionaryLearning(n_components=self.D, alpha=1, n_iter=500)
        self.X = dico.fit(self.dictionary_learn_images).components_
        self.X = self.X.T
        dt = time() - t0
        print('done in %.2fs.' % dt)

        coder = SparseCoder(dictionary=self.X.T, transform_algorithm='omp')

        self.beta_train = coder.transform(self.y_train)
        self.beta_validation = coder.transform(self.y_validation)


    def generate_random_design_matrix(self):
        self.X = np.random.randn(self.D, self.beta_train.shape[1])

        self.y_train = np.dot(self.beta_train, self.X.T)
        self.y_validation = np.dot(self.beta_validation, self.X.T)

    def plot_learnt_dictionary(self):
        for i, comp in enumerate(self.X[:10]):
            plt.imshow(comp.reshape(self.image_size), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.show()


if __name__=='__main__':
    data=MnistDataSparseCoding(D=400, train_size=100, valid_size=100, image_size=10)
    data.check_download()
    data.learn_dictionary()
    data.plot_learnt_dictionary()
