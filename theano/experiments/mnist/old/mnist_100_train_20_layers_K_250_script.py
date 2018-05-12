import numpy as np
import six.moves.cPickle as pickle
from tqdm import tqdm

from comparator.compare_sequential import SequentialComparator
from compare_mnist.mnist_data import MnistData


class MnistSequentialComparator_DFGHJDF(SequentialComparator):
    def __init__(self, K, L, learning_rate=0.0001, n_train_sample=1, n_validation_sample=1,
                 train_freq=True, train_bayes=True, train_shared_bayes=True, save_history=False):

        self.data = MnistData(K=K)
        self.data.check_download(normalise=False)
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


    lista_time = comparator.recorders['lista'].time
    sh_bayes_time = comparator.recorders['shared_bayes'].time
    lista_nmse = comparator.recorders['lista'].validation_loss
    lista_f_meas = comparator.recorders['lista'].validation_f_meas
    sh_bayes_nmse = comparator.recorders['shared_bayes'].validation_loss
    sh_bayes_f_meas = comparator.recorders['shared_bayes'].validation_f_meas

    np.savez('time_100_train_20_layers_K_250', lista_time=lista_time, sh_bayes_time=sh_bayes_time, lista_nmse=lista_nmse, lista_f_meas=lista_f_meas, sh_bayes_nmse=sh_bayes_nmse, sh_bayes_f_meas=sh_bayes_f_meas)


    comparator.plot_quality_history()

    with open('mnist_100_train_20_layers_K_250.pkl', 'wb') as f:
        pickle.dump(comparator, f)

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
