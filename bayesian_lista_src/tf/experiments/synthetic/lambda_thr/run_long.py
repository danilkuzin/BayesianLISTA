import os

import numpy as np
from tqdm import tqdm, trange
import tensorflow as tf

from tf.comparator.compare_sequential import SequentialComparator
from tf.data.synthetic.data_generator import DataGenerator, SyntheticData
from tf.experiments.synthetic.experiments_parameters import load_long_experiment, load_quick_experiment

tf.enable_eager_execution()

rseed, D, K, L, batch_size, validation_size, n_iter = load_long_experiment()

lambda_array = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]

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


for rseed in range(10):
    np.random.seed(rseed)
    tf.random.set_random_seed(rseed)
    data_generator = DataGenerator(D, K)
    beta_train, y_train, _ = data_generator.new_sample(batch_size)
    beta_validation, y_validation, _ = data_generator.new_sample(batch_size)
    data = SyntheticData(data_generator.X, beta_train, y_train, beta_validation, y_validation)

    for i, lam in enumerate(tqdm(lambda_array)):

        comparator = SequentialComparator(D, K, L, learning_rate=0.0001, data=data, train_freq=True,
                                          train_shared_bayes=True, use_ista=True, use_fista=True, save_history=False,
                                          initial_lambda=lam)
        for _ in trange(n_iter):
            comparator.train_iteration()

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

