import errno
import numpy as np
import six.moves.cPickle as pickle

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import os

from compare_freq_bayes.compare_sequential_with_ista_fista import SequentialComparatorIstaFista
from experiments.synthetic.experiments_parameters import load_long_experiment
from compare_mnist.mnist_data import MnistData

class MnistSequentialComparatorIstaFista(SequentialComparatorIstaFista):
    def __init__(self, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1,
                 train_freq=False, train_bayes=False, train_shared_bayes=False, use_ista=True, use_fista=True, save_history=False):

        self.data = MnistData(K=K)
        self.data.check_download(normalise=False)
        self.data.learn_dictionary()
        self.D = self.data.train_data.shape[1]

        super().__init__(self.D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, use_ista, use_fista, save_history)

        self.beta_train = self.data.train_data
        self.y_train = self.data.y_train
        self.beta_validation = self.data.validation_data
        self.y_validation = self.data.y_validation


if __name__ == '__main__':
    try:
        os.makedirs('ista_fista')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    np.random.seed(1)

    K = 100
    L = 20

    comparator = MnistSequentialComparatorIstaFista(K, L, learning_rate=0.0001, n_train_sample=100,
                                      n_validation_sample=100, train_freq=False,
                                      train_bayes=False, train_shared_bayes=False, use_ista=True, use_fista=True, save_history=True)
    n_iter = 500

    for _ in tqdm(range(n_iter)):
        comparator.train_iteration()

    comparator.plot_quality_history()


    comparator.save_numpy('ista_fista/results_{}_K'.format(K))
