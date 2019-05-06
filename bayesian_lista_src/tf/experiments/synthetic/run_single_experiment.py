import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from tf.comparator.compare_sequential import SequentialComparator
from tf.data.synthetic.data_generator import DataGenerator, SyntheticData
from tf.experiments.synthetic.experiments_parameters import load_long_experiment, load_quick_experiment

import matplotlib.pyplot as plt
tf.enable_eager_execution()

rseed, D, K, L, batch_size, validation_size, n_iter = load_quick_experiment()
n_iter = 2

def run_single_experiment(rseed, D, K, L, batch_size, n_iter):
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

    return [(recorder_name, recorder.get_metrics()) for recorder_name, recorder in comparator.recorders.items()]


results = run_single_experiment(rseed, D, K, L, batch_size, n_iter)

metric_keys = results[0][1].keys()
for key in metric_keys:
    for result_name, result_metrics in results:
        plt.plot(result_metrics[key], label=result_name)
    plt.title(key)
    plt.legend()
    plt.show()
