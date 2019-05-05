import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from tf.comparator.compare_sequential import SequentialComparator
from tf.data.synthetic.data_generator import DataGenerator, SyntheticData
from tf.experiments.synthetic.experiments_parameters import load_long_experiment

tf.enable_eager_execution()

rseed, D, _, L, batch_size, validation_size, n_iter = load_long_experiment()
K_array = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

freq_train_loss = np.zeros((len(K_array), n_iter))
freq_validation_loss = np.zeros((len(K_array), n_iter))
freq_train_f_measure = np.zeros((len(K_array), n_iter))
freq_validation_f_measure = np.zeros((len(K_array), n_iter))
freq_time = np.zeros((len(K_array), n_iter))

sh_bayes_train_loss = np.zeros((len(K_array), n_iter))
sh_bayes_validation_loss = np.zeros((len(K_array), n_iter))
sh_bayes_train_f_measure = np.zeros((len(K_array), n_iter))
sh_bayes_validation_f_measure = np.zeros((len(K_array), n_iter))
sh_bayes_time = np.zeros((len(K_array), n_iter))

ista_train_loss = np.zeros((len(K_array), n_iter))
ista_validation_loss = np.zeros((len(K_array), n_iter))
ista_train_f_measure = np.zeros((len(K_array), n_iter))
ista_validation_f_measure = np.zeros((len(K_array), n_iter))

fista_train_loss = np.zeros((len(K_array), n_iter))
fista_validation_loss = np.zeros((len(K_array), n_iter))
fista_train_f_measure = np.zeros((len(K_array), n_iter))
fista_validation_f_measure = np.zeros((len(K_array), n_iter))

for rseed in range(10):
    np.random.seed(rseed)
    tf.random.set_random_seed(rseed)

    for i, K in enumerate(tqdm(K_array)):
        data_generator = DataGenerator(D, K)
        beta_train, y_train, _ = data_generator.new_sample(batch_size)
        beta_validation, y_validation, _ = data_generator.new_sample(batch_size)
        data = SyntheticData(data_generator.X, beta_train, y_train, beta_validation, y_validation)

        comparator = SequentialComparator(D, K, L, learning_rate=0.0001, data=data, train_freq=True,
                                          train_shared_bayes=True, use_ista=True, use_fista=True, save_history=False,
                                          initial_lambda=0.1)
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
    file_name = path_name + 'undersampling_measures'
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

