import numpy as np
import tensorflow as tf
from tqdm import trange

from tf.experiments.active_learning.experiment import ActiveLearningExperiments

for rseed in trange(10):
    np.random.seed(rseed)
    tf.set_random_seed(rseed)

    n_upd_iter = 10
    update_size = 1
    n_active_iter = 50

    active_learning_experiments = ActiveLearningExperiments(update_size=update_size)
    active_learning_experiments.get_synthetic_data_active_learning()
    active_learning_experiments.init_and_pretrain_lista()

    for i in trange(n_upd_iter):
        active_learning_experiments.choose_next_random_from_pool()
        active_learning_experiments.choose_next_train_active_from_pool()
        for j in range(n_active_iter):
            active_learning_experiments.learning_iter()
        active_learning_experiments.update_quality()

    freq_validation_loss = active_learning_experiments.freq_validation_loss
    non_active_bayes_validation_loss = active_learning_experiments.non_active_shared_bayesian_validation_loss
    active_bayes_validation_loss = active_learning_experiments.active_bayesian_validation_loss

    freq_validation_f_measure = active_learning_experiments.freq_validation_f_meas
    non_active_bayes_validation_f_measure = active_learning_experiments.non_active_shared_bayesian_validation_f_meas
    active_bayes_validation_f_measure = active_learning_experiments.active_bayesian_validation_f_meas

    np.savez('synthetic_active_rseed_{}'.format(rseed), freq_validation_loss=freq_validation_loss,
             non_active_bayes_validation_loss=non_active_bayes_validation_loss,
             active_bayes_validation_loss=active_bayes_validation_loss,
             freq_validation_f_measure=freq_validation_f_measure,
             non_active_bayes_validation_f_measure=non_active_bayes_validation_f_measure,
             active_bayes_validation_f_measure=active_bayes_validation_f_measure)
