from compare_freq_bayes.compare_sequential_selected_algorithms import SequentialComparatorWithAlgotithmSelection
import numpy as np
import matplotlib.pyplot as plt

class SequentialComparatorWithWeightsCheck(SequentialComparatorWithAlgotithmSelection):
    def __init__(self, D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes):
        super().__init__(D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes)
        self.freq_w0_hist = []
        self.shared_bayes_w0_hist = []
        self.shared_bayes_w0_var_hist = []
        self.freq_s0_hist = []
        self.shared_bayes_s0_hist = []
        self.shared_bayes_s0_var_hist = []

    def get_weights(self):
        self.freq_w0_hist.append(self.freq_lista.lista.W.get_value())
        self.shared_bayes_w0_hist.append(self.shared_bayesian_lista.pbp_instance.network.params_W_M.get_value())
        self.shared_bayes_w0_var_hist.append(self.shared_bayesian_lista.pbp_instance.network.params_W_V.get_value())
        self.freq_s0_hist.append(self.freq_lista.lista.S.get_value())
        self.shared_bayes_s0_hist.append(self.shared_bayesian_lista.pbp_instance.network.params_S_M.get_value())
        self.shared_bayes_s0_var_hist.append(self.shared_bayesian_lista.pbp_instance.network.params_S_V.get_value())


if __name__ == '__main__':

    np.random.seed(1)

    D = 100
    K = 50
    L = 4

    batch_size = 500
    validation_size = 10

    n_iter = 100

    comparator = SequentialComparatorWithWeightsCheck(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                                      n_validation_sample=validation_size, train_freq=True, train_bayes=False,
                                                      train_shared_bayes=True)


    for _ in range(n_iter):
        comparator.get_weights()
        comparator.train_iteration()

    comparator.freq_w0_hist = np.array(comparator.freq_w0_hist)
    comparator.shared_bayes_w0_hist = np.array(comparator.shared_bayes_w0_hist)
    comparator.shared_bayes_w0_var_hist = np.array(comparator.shared_bayes_w0_var_hist)

    plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_w0_hist[:, 0, 0], label="freq w[0, 0]")
    plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_w0_hist[:, 0, 0], label="bayes w[0, 0]")

    lower = comparator.shared_bayes_w0_hist[:, 0, 0] - 2 * np.sqrt(comparator.shared_bayes_w0_var_hist[:, 0, 0])
    upper = comparator.shared_bayes_w0_hist[:, 0, 0] + 2 * np.sqrt(comparator.shared_bayes_w0_var_hist[:, 0, 0])
    plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')

    plt.legend()
    plt.show()

    comparator.freq_s0_hist = np.array(comparator.freq_s0_hist)
    comparator.shared_bayes_s0_hist = np.array(comparator.shared_bayes_s0_hist)
    comparator.shared_bayes_s0_var_hist = np.array(comparator.shared_bayes_s0_var_hist)

    plt.plot(np.linspace(0, n_iter, n_iter), comparator.freq_s0_hist[:, 0, 0], label="freq s[0, 0]")
    plt.plot(np.linspace(0, n_iter, n_iter), comparator.shared_bayes_s0_hist[:, 0, 0], label="bayes s[0, 0]")

    lower = comparator.shared_bayes_s0_hist[:, 0, 0] - 2 * np.sqrt(comparator.shared_bayes_s0_var_hist[:, 0, 0])
    upper = comparator.shared_bayes_s0_hist[:, 0, 0] + 2 * np.sqrt(comparator.shared_bayes_s0_var_hist[:, 0, 0])
    plt.fill_between(np.linspace(0, n_iter, n_iter), lower, upper, color='aquamarine', edgecolor='blue')

    plt.legend()
    plt.show()