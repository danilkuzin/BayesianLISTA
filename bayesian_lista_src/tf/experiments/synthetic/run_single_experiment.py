import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from tf.comparator.compare_sequential import SequentialComparator
from tf.data.synthetic.data_generator import DataGenerator, SyntheticData
from tf.experiments.synthetic.experiments_parameters import load_long_experiment

tf.enable_eager_execution()

rseed, D, K, L, batch_size, validation_size, n_iter = load_long_experiment()

def run_single_experiment(rseed,  D, K, L, batch_size, validation_size, n_iter)
    np.random.seed(rseed)
    tf.random.set_random_seed(rseed)

    data_generator = DataGenerator(D, K)
    beta_train, y_train, _ = data_generator.new_sample(batch_size)
    beta_validation, y_validation, _ = data_generator.new_sample(batch_size)
    data = SyntheticData(data_generator.X, beta_train, y_train, beta_validation, y_validation)

    comparator = SequentialComparator(D, K, L, learning_rate=0.0001, data=data, train_freq=True,
                                      train_shared_bayes=True, use_ista=True, use_fista=True, save_history=False,
                                      initial_lambda=0.1)
    for _ in trange(n_iter):
        comparator.train_iteration()

    return [recorder.get_metrics() for recorder in comparator.recorders]

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