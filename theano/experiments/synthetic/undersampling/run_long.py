import numpy as np
import six.moves.cPickle as pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from compare_freq_bayes.compare_sequential import SequentialComparator
from experiments.synthetic.experiments_parameters import load_long_experiment

rseed, D, _, L, batch_size, validation_size, n_iter = load_long_experiment()
np.random.seed(rseed)
K_array = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

freq_train_loss = np.zeros((len(K_array), n_iter))
freq_validation_loss = np.zeros((len(K_array), n_iter))
freq_train_f_measure = np.zeros((len(K_array), n_iter))
freq_validation_f_measure = np.zeros((len(K_array), n_iter))
sh_bayes_train_loss = np.zeros((len(K_array), n_iter))
sh_bayes_validation_loss = np.zeros((len(K_array), n_iter))
sh_bayes_train_f_measure = np.zeros((len(K_array), n_iter))
sh_bayes_validation_f_measure = np.zeros((len(K_array), n_iter))

for i, K in enumerate(tqdm(K_array)):
    comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                      n_validation_sample=validation_size, train_freq=True, train_bayes=False,
                                      train_shared_bayes=True)
    for _ in trange(n_iter):
        comparator.train_iteration()

    freq_train_loss[i] = comparator.freq_train_loss
    freq_validation_loss[i] = comparator.freq_validation_loss
    freq_train_f_measure[i] = comparator.freq_train_f_meas
    freq_validation_f_measure[i] = comparator.freq_validation_f_meas

    sh_bayes_train_loss[i] = comparator.shared_bayesian_train_loss
    sh_bayes_validation_loss[i] = comparator.shared_bayesian_validation_loss
    sh_bayes_train_f_measure[i] = comparator.shared_bayesian_train_f_meas
    sh_bayes_validation_f_measure[i] = comparator.shared_bayesian_validation_f_meas

    with open('undersampling_{}.pkl'.format(K), 'wb') as f:
        pickle.dump(comparator, f)


np.savez('undersampling_measures', freq_train_loss=freq_train_loss, freq_validation_loss=freq_validation_loss,
         freq_train_f_measure=freq_train_f_measure, freq_validation_f_measure=freq_validation_f_measure,
         sh_bayes_train_loss=sh_bayes_train_loss, sh_bayes_validation_loss=sh_bayes_validation_loss,
         sh_bayes_train_f_measure=sh_bayes_train_f_measure, sh_bayes_validation_f_measure=sh_bayes_validation_f_measure)


plt.plot(K_array, freq_train_loss[:, -1], label="freq train")
plt.plot(K_array, freq_validation_loss[:, -1], label="freq valid")
plt.plot(K_array, sh_bayes_train_loss[:, -1], label="Bayes train")
plt.plot(K_array, sh_bayes_validation_loss[:, -1], label="Bayes valid")

plt.legend()
plt.savefig('compare_nmse_vs_undersampling.eps', format='eps')
plt.show()

plt.plot(K_array, freq_train_f_measure[:, -1], label="freq train")
plt.plot(K_array, freq_validation_f_measure[:, -1], label="freq valid")
plt.plot(K_array, sh_bayes_train_f_measure[:, -1], label="Bayes train")
plt.plot(K_array, sh_bayes_validation_f_measure[:, -1], label="Bayes valid")

plt.legend()
plt.savefig('compare_f_measure_vs_undersampling.eps', format='eps')
plt.show()

