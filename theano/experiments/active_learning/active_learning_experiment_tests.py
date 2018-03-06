import unittest

from experiments.active_learning.active_learning_experiment import get_synthetic_data_active_learning, \
    get_mnist_data_active_learning


class TestActiveLearningExperiments():
    def get_synthetic_data_test(self):
        experiments = ActiveLearningExperiments()
        train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y = experiments.get_synthetic_data_active_learning()
        print(train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y)

    def get_mnist_data(self):
        train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y = get_mnist_data_active_learning()
        print(train_data_beta, train_data_y, pool_data_beta, pool_data_y, test_data_beta, test_data_y)


if __name__ == '__main__':
    tests = TestActiveLearningExperiments()
    tests.get_synthetic_data_test()
    tests.get_mnist_data()