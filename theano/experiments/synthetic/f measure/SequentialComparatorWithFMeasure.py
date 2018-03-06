from compare_freq_bayes.compare_sequential_selected_algorithms import SequentialComparatorWithAlgotithmSelection
import numpy as np
import matplotlib.pyplot as plt

from experiments.synthetic.experiments_parameters import load_synthetic_experiment_1


class SequentialComparatorWithFMeasure(SequentialComparatorWithAlgotithmSelection):
    def __init__(self, D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes):
        super().__init__(D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes)

        if self.train_freq:
            self.freq_train_f_meas = []
            self.freq_validation_f_meas = []

        if self.train_bayes:
            self.bayesian_train_f_meas = []
            self.bayesian_validation_f_meas = []

        if self.train_shared_bayes:
            self.shared_bayesian_train_f_meas = []
            self.shared_bayesian_validation_f_meas = []

    def train_iteration(self):
        if self.train_freq:
            cur_freq_train_loss, self.freq_lista.train_iteration_nmse(beta_train=self.beta_train, y_train=self.y_train)
            self.freq_train_loss.append(

        if self.train_bayes:
            self.bayesian_train_loss.append(
                self.bayesian_lista.train_iteration_nmse(beta_train=self.beta_train, y_train=self.y_train))
        if self.train_shared_bayes:
            self.shared_bayesian_train_loss.append(
                self.shared_bayesian_lista.train_iteration_nmse(beta_train=self.beta_train, y_train=self.y_train))

        if self.train_freq:
            self.freq_validation_loss.append(
                self.freq_lista.test_nmse(beta_test=self.beta_validation, y_test=self.y_validation))
        if self.train_bayes:
            self.bayesian_validation_loss.append(
                self.bayesian_lista.test_nmse(beta_test=self.beta_validation, y_test=self.y_validation))
        if self.train_shared_bayes:
            self.shared_bayesian_validation_loss.append(
                self.shared_bayesian_lista.test_nmse(beta_test=self.beta_validation, y_test=self.y_validation))

if __name__ == '__main__':

    rseed, D, K, L, batch_size, validation_size, n_iter = load_synthetic_experiment_1()
    np.random.seed(rseed)

    comparator = SequentialComparatorWithFMeasure(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
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