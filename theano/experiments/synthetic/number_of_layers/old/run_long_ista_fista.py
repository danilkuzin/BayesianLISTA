import errno
import numpy as np
import six.moves.cPickle as pickle

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import os

from compare_freq_bayes.compare_sequential_with_ista_fista import SequentialComparatorIstaFista
from experiments.synthetic.experiments_parameters import load_long_experiment

rseed, D, K, _, batch_size, validation_size, n_iter = load_long_experiment()
np.random.seed(rseed)
L_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]

freq_train_loss = np.zeros((len(L_array), n_iter))
freq_validation_loss = np.zeros((len(L_array), n_iter))
freq_train_f_measure = np.zeros((len(L_array), n_iter))
freq_validation_f_measure = np.zeros((len(L_array), n_iter))
sh_bayes_train_loss = np.zeros((len(L_array), n_iter))
sh_bayes_validation_loss = np.zeros((len(L_array), n_iter))
sh_bayes_train_f_measure = np.zeros((len(L_array), n_iter))
sh_bayes_validation_f_measure = np.zeros((len(L_array), n_iter))

try:
    os.makedirs('ista_fista')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for rseed in trange(10):
    np.random.seed(rseed)
    for i, L in enumerate(tqdm(L_array)):
        comparator = SequentialComparatorIstaFista(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                          n_validation_sample=validation_size, train_freq=False,
                                          train_bayes=False, train_shared_bayes=False, use_ista=True, use_fista=True)
        for _ in range(n_iter):
            comparator.train_iteration()


        comparator.save_numpy('ista_fista/fista_ista_{}_L_{}_rseed'.format(L, rseed))
