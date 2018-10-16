from comparator.compare_sequential import SequentialComparator
from compare_mnist.mnist_data import MnistData


class MnistSequentialComparator(SequentialComparator):
    def __init__(self, K, L, learning_rate, n_train_sample, n_validation_sample,
                 train_freq, train_bayes, train_shared_bayes, use_ista, use_fista, save_history, initial_lambda,
                 learnt_dictionary, normalise_dictionary):

        data = MnistData(K=K, train_size=n_train_sample, valid_size=n_validation_sample)
        data.check_download(normalise=False)
        if learnt_dictionary:
            data.learn_dictionary()
        else:
            data.random_dictionary(normalise=normalise_dictionary)

        D = data.beta_train.shape[1]

        super().__init__(D=D, K=K, L=L, data=data, learning_rate=learning_rate, train_freq=train_freq,
                         train_bayes=train_bayes, train_shared_bayes=train_shared_bayes, use_ista=use_ista,
                         use_fista=use_fista, save_history=save_history, initial_lambda=initial_lambda)

