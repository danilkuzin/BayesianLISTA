import numpy as np
import six.moves.cPickle as pickle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
from comparator.compare_sequential import SequentialComparator
from experiments.synthetic_overcomplete.experiments_parameters import load_long_experiment

rseed, D, K, L, batch_size, validation_size, n_iter = load_long_experiment()
np.random.seed(rseed)
lambda_array = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

freq_train_loss = np.zeros((len(lambda_array), n_iter))
freq_validation_loss = np.zeros((len(lambda_array), n_iter))
freq_train_f_measure = np.zeros((len(lambda_array), n_iter))
freq_validation_f_measure = np.zeros((len(lambda_array), n_iter))
freq_time = np.zeros((len(lambda_array), n_iter))

sh_bayes_train_loss = np.zeros((len(lambda_array), n_iter))
sh_bayes_validation_loss = np.zeros((len(lambda_array), n_iter))
sh_bayes_train_f_measure = np.zeros((len(lambda_array), n_iter))
sh_bayes_validation_f_measure = np.zeros((len(lambda_array), n_iter))
sh_bayes_time = np.zeros((len(lambda_array), n_iter))

ista_train_loss = np.zeros((len(lambda_array), n_iter))
ista_validation_loss = np.zeros((len(lambda_array), n_iter))
ista_train_f_measure = np.zeros((len(lambda_array), n_iter))
ista_validation_f_measure = np.zeros((len(lambda_array), n_iter))

fista_train_loss = np.zeros((len(lambda_array), n_iter))
fista_validation_loss = np.zeros((len(lambda_array), n_iter))
fista_train_f_measure = np.zeros((len(lambda_array), n_iter))
fista_validation_f_measure = np.zeros((len(lambda_array), n_iter))

for rseed in range(1):
    np.random.seed(rseed)
    for i, lam in enumerate(tqdm(lambda_array)):
        comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                          n_validation_sample=validation_size, train_freq=True, train_bayes=False,
                                          train_shared_bayes=True, use_ista=True, use_fista=True, save_history=False,
                                          initial_lambda=lam)
        for _ in trange(n_iter):
            comparator.train_iteration()

        # comparator.save_numpy('undersampling_{}'.format(K))

        freq_train_loss[i] = comparator.recorders['lista'].train_loss
        freq_validation_loss[i] = comparator.recorders['lista'].validation_loss
        freq_train_f_measure[i] = comparator.recorders['lista'].train_f_meas
        freq_validation_f_measure[i] = comparator.recorders['lista'].validation_f_meas
        freq_time = comparator.recorders['lista'].time

        sh_bayes_train_loss[i] = comparator.recorders['shared_bayes'].train_loss
        sh_bayes_validation_loss[i] = comparator.recorders['shared_bayes'].validation_loss
        sh_bayes_train_f_measure[i] = comparator.recorders['shared_bayes'].train_f_meas
        sh_bayes_validation_f_measure[i] = comparator.recorders['shared_bayes'].validation_f_meas
        sh_bayes_time = comparator.recorders['shared_bayes'].time

        ista_train_loss[i] = comparator.recorders['ista'].train_loss
        ista_validation_loss[i] = comparator.recorders['ista'].validation_loss
        ista_train_f_measure[i] = comparator.recorders['ista'].train_f_meas
        ista_validation_f_measure[i] = comparator.recorders['ista'].validation_f_meas

        fista_train_loss[i] = comparator.recorders['fista'].train_loss
        fista_validation_loss[i] = comparator.recorders['fista'].validation_loss
        fista_train_f_measure[i] = comparator.recorders['fista'].train_f_meas
        fista_validation_f_measure[i] = comparator.recorders['fista'].validation_f_meas

    path_name = '{}/'.format(rseed)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    file_name = path_name + 'lambda_measures'
    np.savez(file_name, freq_train_loss=freq_train_loss, freq_validation_loss=freq_validation_loss,
             freq_train_f_measure=freq_train_f_measure, freq_validation_f_measure=freq_validation_f_measure, freq_time=freq_time,
             sh_bayes_train_loss=sh_bayes_train_loss, sh_bayes_validation_loss=sh_bayes_validation_loss,
             sh_bayes_train_f_measure=sh_bayes_train_f_measure, sh_bayes_validation_f_measure=sh_bayes_validation_f_measure, sh_bayes_time=sh_bayes_time,
             ista_train_loss=ista_train_loss, ista_validation_loss=ista_validation_loss, ista_train_f_measure=ista_train_f_measure, ista_validation_f_measure=ista_validation_f_measure,
             fista_train_loss=fista_train_loss, fista_validation_loss=fista_validation_loss, fista_train_f_measure=fista_train_f_measure, fista_validation_f_measure=fista_validation_f_measure)


# plt.plot(K_array, freq_train_loss[:, -1], label="freq train")
# plt.plot(K_array, freq_validation_loss[:, -1], label="freq valid")
# plt.plot(K_array, sh_bayes_train_loss[:, -1], label="Bayes train")
# plt.plot(K_array, sh_bayes_validation_loss[:, -1], label="Bayes valid")
#
# plt.legend()
# plt.savefig('compare_nmse_vs_undersampling.eps', format='eps')
# plt.show()
#
# plt.plot(K_array, freq_train_f_measure[:, -1], label="freq train")
# plt.plot(K_array, freq_validation_f_measure[:, -1], label="freq valid")
# plt.plot(K_array, sh_bayes_train_f_measure[:, -1], label="Bayes train")
# plt.plot(K_array, sh_bayes_validation_f_measure[:, -1], label="Bayes valid")
#
# plt.legend()
# plt.savefig('compare_f_measure_vs_undersampling.eps', format='eps')
# plt.show()

