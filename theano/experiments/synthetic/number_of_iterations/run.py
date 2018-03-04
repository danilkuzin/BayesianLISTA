from compare_freq_bayes.compare_sequential import SequentialComparator
import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

from compare_freq_bayes.compare_sequential_selected_algorithms import SequentialComparatorWithAlgotithmSelection

np.random.seed(10)

D = 10
K = 8
L = 4

n_iter_array = [1, 5, 10, 2000, 4000, 6000, 8000, 10000]

batch_size = 1000
validation_size = 100

freq_train_loss = np.zeros(len(n_iter_array))
freq_validation_loss = np.zeros(len(n_iter_array))

for i, n_iter in enumerate(n_iter_array):
    print('iteration {}'.format(n_iter))
    print('compiling...')
    comparator = SequentialComparatorWithAlgotithmSelection(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                      n_validation_sample=validation_size, train_freq=True, train_bayes=False, train_shared_bayes=False)
    print('training...')

    for ttt in range(n_iter):
        learning_rate = (1 / (ttt + 1)) * 0.0001
        comparator.train_iteration(learning_rate)


    freq_train_loss[i] = comparator.freq_train_loss[-1]
    freq_validation_loss[i] = comparator.freq_validation_loss[-1]

    with open('number_of_iterations_freq_{}.pkl'.format(L), 'wb') as f:
        pickle.dump(comparator, f)


plt.plot(n_iter_array, freq_train_loss, label="freq train loss")
plt.plot(n_iter_array, freq_validation_loss, label="freq valid loss")

plt.legend()
plt.show()

