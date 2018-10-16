import numpy as np
import six.moves.cPickle as pickle
from tqdm import tqdm

from compare_freq_bayes.compare_sequential import SequentialComparator
from compare_mnist.mnist_data import MnistData
import tensorflow as tf


class MnistSequentialComparator_DFGHJDF(SequentialComparator):
    def __init__(self, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1,
                 train_freq=True, train_bayes=True, train_shared_bayes=True, save_history=False):

        self.data = MnistData(K=K)
        self.data.check_download()
        self.data.learn_dictionary()
        self.D = self.data.train_data.shape[1]

        super().__init__(self.D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, save_history)

        self.beta_train = self.data.train_data
        self.y_train = self.data.y_train
        self.beta_validation = self.data.validation_data
        self.y_validation = self.data.y_validation

if __name__ == '__main__':

    np.random.seed(1)
    tf.set_random_seed(1234)

    K = 250
    L = 20

    # batch_size = 5000
    # validation_size = 100

    saved_comparator_file_name = []#'test_S_convergence.pkl'


    if not saved_comparator_file_name:
        comparator = MnistSequentialComparator_DFGHJDF(K, L, learning_rate=0.0001,
                                                       train_bayes=False, save_history=True,
                                                       n_train_sample=100, n_validation_sample=100)
    else:
        comparator = pickle.load(open(saved_comparator_file_name, 'rb'))




    n_iter = 500

    for _ in tqdm(range(n_iter)):
       comparator.train_iteration()

    comparator.plot_quality_history()

    freq_train_loss = comparator.freq_train_loss
    freq_validation_loss = comparator.freq_validation_loss
    shared_bayesian_train_loss = comparator.shared_bayesian_train_loss
    shared_bayesian_validation_loss = comparator.shared_bayesian_validation_loss
    freq_train_f_measure = comparator.freq_train_f_meas
    freq_validation_f_measure = comparator.freq_validation_f_meas
    shared_bayesian_f_measure = comparator.shared_bayesian_train_f_meas
    shared_bayesian_train_f_measure = comparator.shared_bayesian_train_f_meas
    shared_bayesian_validation_f_measure = comparator.shared_bayesian_validation_f_meas
    freq_beta_train_est = comparator.freq_lista.predict(comparator.data.y_train)
    freq_beta_validation_est = comparator.freq_lista.predict(comparator.data.y_validation)
    shared_beta_train_est = comparator.shared_bayesian_lista.predict(comparator.data.y_train)
    shared_beta_validation_est = comparator.shared_bayesian_lista.predict(comparator.data.y_validation)
    true_beta_train = comparator.data.train_data
    true_beta_validation = comparator.data.validation_data
    y_train = comparator.data.y_train
    y_validation = comparator.data.y_validation
    D = comparator.data.train_data.shape[1]
    train_size = comparator.data.train_data.shape[0]
    bayes_W_M = comparator.shared_bayesian_lista.pbp_instance.network.params_W_M.get_value()
    bayes_W_V = comparator.shared_bayesian_lista.pbp_instance.network.params_W_V.get_value()
    bayes_S_M = comparator.shared_bayesian_lista.pbp_instance.network.params_S_M.get_value()
    bayes_S_V = comparator.shared_bayesian_lista.pbp_instance.network.params_S_V.get_value()
    np.savez('mnist_100_train_20_layers_K_250_bayes_weights', D=D, K=K, L=L, bayes_S_M=bayes_S_M, bayes_S_V=bayes_S_V,
             bayes_W_M=bayes_W_M, bayes_W_V=bayes_W_V)
    np.savez('mnist_100_train_20_layers_K_250_beta_est', D=D, K=K, L=L, freq_beta_train_est=freq_beta_train_est,
             freq_beta_validation_est=freq_beta_validation_est, shared_beta_train_est=shared_beta_train_est,
             shared_beta_validation_est=shared_beta_validation_est, true_beta_train=true_beta_train,
             true_beta_validation=true_beta_validation, y_train=y_train, y_validation=y_validation)
    np.savez('mnist_100_train_20_layers_K_250_quality', D=D, K=K, L=L, freq_train_f_measure=freq_train_f_measure,
             freq_train_loss=freq_train_loss, freq_validation_f_measure=freq_validation_f_measure,
             freq_validation_loss=freq_validation_loss, shared_bayesian_train_f_measure=shared_bayesian_train_f_measure,
             shared_bayesian_train_loss=shared_bayesian_train_loss,
             shared_bayesian_validation_f_measure=shared_bayesian_validation_f_measure,
             shared_bayesian_validation_loss=shared_bayesian_validation_loss)
    np.savez('mnist_100_train_20_layers_K_250_params', D=D, K=K, L=L, train_size=train_size)


    #with open('mnist_100_train_20_layers_K_250.pkl', 'wb') as f:
    #    pickle.dump(comparator, f)

    # comparator.freq_w_hist = np.array(comparator.freq_w_hist)
    # comparator.shared_bayes_w_hist = np.array(comparator.shared_bayes_w_hist)
    # comparator.shared_bayes_w_var_hist = np.array(comparator.shared_bayes_w_var_hist)
    #
    # i1 = 0
    # i2 = 0
    #
    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_w_hist[:, i1, i2], label="freq w[0, 0]")
    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_w_hist[:, i1, i2], label="bayes w[0, 0]")
    #
    # lower = comparator.shared_bayes_w_hist[:, i1, i2] - 2 * np.sqrt(comparator.shared_bayes_w_var_hist[:, i1, i2])
    # upper = comparator.shared_bayes_w_hist[:, i1, i2] + 2 * np.sqrt(comparator.shared_bayes_w_var_hist[:, i1, i2])
    # plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')
    #
    # plt.legend()
    # plt.show()
    #
    # comparator.freq_s_hist = np.array(comparator.freq_s_hist)
    # comparator.shared_bayes_s_hist = np.array(comparator.shared_bayes_s_hist)
    # comparator.shared_bayes_s_var_hist = np.array(comparator.shared_bayes_s_var_hist)
    #
    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_s_hist[:, i1, i2], label="freq s[0, 0]")
    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_s_hist[:, i1, i2], label="bayes s[0, 0]")
    #
    # lower = comparator.shared_bayes_s_hist[:, i1, i2] - 2 * np.sqrt(comparator.shared_bayes_s_var_hist[:, i1, i2])
    # upper = comparator.shared_bayes_s_hist[:, i1, i2] + 2 * np.sqrt(comparator.shared_bayes_s_var_hist[:, i1, i2])
    # plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')
    #
    # plt.legend()
    # plt.show()
