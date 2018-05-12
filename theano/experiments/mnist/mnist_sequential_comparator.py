from comparator.compare_sequential import SequentialComparator
from compare_mnist.mnist_data import MnistData


class MnistSequentialComparator(SequentialComparator):
    def __init__(self, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, use_ista, use_fista, save_history, initial_lambda):

        self.data = MnistData(K=K, train_size=n_train_sample, valid_size=n_validation_sample)
        self.data.check_download(normalise=False)
        self.data.learn_dictionary()
        self.D = self.data.train_data.shape[1]

        super().__init__(self.D, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, use_ista, use_fista, save_history, initial_lambda)

        self.beta_train = self.data.train_data
        self.y_train = self.data.y_train
        self.beta_validation = self.data.validation_data
        self.y_validation = self.data.y_validation