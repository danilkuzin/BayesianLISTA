from comparator.compare_sequential import SequentialComparator
from experiments.mnist_sparse_coding.mnist_data_sparse_coding import MnistDataSparseCoding


class MnistSequentialComparatorSparseCoding(SequentialComparator):
    def __init__(self, D, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, use_ista, use_fista, save_history, initial_lambda):

        data = MnistDataSparseCoding(D=D, train_size=n_train_sample, valid_size=n_validation_sample, image_size=10)
        data.check_download()
        data.learn_dictionary()
        K = data.y_train.shape[1]

        super().__init__(D=D, K=K, L=L, data=data, learning_rate=learning_rate, train_freq=train_freq,
                         train_bayes=train_bayes, train_shared_bayes=train_shared_bayes, use_ista=use_ista,
                         use_fista=use_fista, save_history=save_history, initial_lambda=initial_lambda)

        # self.beta_train = self.data.train_data
        # self.y_train = self.data.y_train
        # self.beta_validation = self.data.validation_data
        # self.y_validation = self.data.y_validation

