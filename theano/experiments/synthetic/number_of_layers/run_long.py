import numpy as np
import six.moves.cPickle as pickle

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from compare_freq_bayes.compare_sequential import SequentialComparator
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

for i, L in enumerate(tqdm(L_array)):
    comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                      n_validation_sample=validation_size, train_freq=True,
                                      train_bayes=False, train_shared_bayes=True)
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

    with open('number_of_layers_{}.pkl'.format(L), 'wb') as f:
        pickle.dump(comparator, f)


np.savez('number_of_layers_measures', freq_train_loss=freq_train_loss, freq_validation_loss=freq_validation_loss,
         freq_train_f_measure=freq_train_f_measure, freq_validation_f_measure=freq_validation_f_measure,
         sh_bayes_train_loss=sh_bayes_train_loss, sh_bayes_validation_loss=sh_bayes_validation_loss,
         sh_bayes_train_f_measure=sh_bayes_train_f_measure, sh_bayes_validation_f_measure=sh_bayes_validation_f_measure)


plt.plot(L_array, freq_train_loss[:, -1], label="freq train")
plt.plot(L_array, freq_validation_loss[:, -1], label="freq valid")
plt.plot(L_array, sh_bayes_train_loss[:, -1], label="Bayes train")
plt.plot(L_array, sh_bayes_validation_loss[:, -1], label="Bayes valid")

plt.legend()
plt.savefig('compare_nmse_vs_number_of_layers.eps', format='eps')
plt.show()


plt.plot(L_array, freq_train_f_measure[:, -1], label="freq train")
plt.plot(L_array, freq_validation_f_measure[:, -1], label="freq valid")
plt.plot(L_array, sh_bayes_train_f_measure[:, -1], label="Bayes train")
plt.plot(L_array, sh_bayes_validation_f_measure[:, -1], label="Bayes valid")

plt.legend()
plt.savefig('compare_f_measure_vs_number_of_layers.eps', format='eps')
plt.show()
