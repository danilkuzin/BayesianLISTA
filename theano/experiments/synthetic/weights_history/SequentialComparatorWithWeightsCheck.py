
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from compare_freq_bayes.compare_sequential import SequentialComparator

if __name__ == '__main__':

    np.random.seed(1)

    D = 100
    K = 50
    L = 4

    batch_size = 500
    validation_size = 10

    n_iter = 1000

    comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                                      n_validation_sample=validation_size, train_freq=False, train_bayes=False,
                                                      train_shared_bayes=True, save_history=True)


    for _ in trange(n_iter):
        comparator.train_iteration()

    # comparator.freq_w_hist = np.array(comparator.freq_w_hist)
    comparator.shared_bayes_w_hist = np.array(comparator.shared_bayes_w_hist)
    comparator.shared_bayes_w_var_hist = np.array(comparator.shared_bayes_w_var_hist)

    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_w_hist[:, 0, 0], label="freq w[0, 0]")
    plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_w_hist[:, 0, 0], label="bayes w[0, 0]")

    lower = comparator.shared_bayes_w_hist[:, 0, 0] - 2 * np.sqrt(comparator.shared_bayes_w_var_hist[:, 0, 0])
    upper = comparator.shared_bayes_w_hist[:, 0, 0] + 2 * np.sqrt(comparator.shared_bayes_w_var_hist[:, 0, 0])
    plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')

    plt.legend()
    plt.show()

    # comparator.freq_s_hist = np.array(comparator.freq_s_hist)
    comparator.shared_bayes_s_hist = np.array(comparator.shared_bayes_s_hist)
    comparator.shared_bayes_s_var_hist = np.array(comparator.shared_bayes_s_var_hist)

    # plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_s_hist[:, 0, 0], label="freq s[0, 0]")
    plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_s_hist[:, 0, 0], label="bayes s[0, 0]")

    lower = comparator.shared_bayes_s_hist[:, 0, 0] - 2 * np.sqrt(comparator.shared_bayes_s_var_hist[:, 0, 0])
    upper = comparator.shared_bayes_s_hist[:, 0, 0] + 2 * np.sqrt(comparator.shared_bayes_s_var_hist[:, 0, 0])
    plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')

    plt.legend()
    plt.show()