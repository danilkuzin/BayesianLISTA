from compare_freq_bayes.compare_sequential import SequentialComparator
import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

from compare_freq_bayes.compare_sequential_selected_algorithms import SequentialComparatorWithAlgotithmSelection

np.random.seed(1)

D = 10
K = 8
L_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]

n_iter = 10000

batch_size = 5000
validation_size = 100

freq_train_loss = np.zeros((len(L_array), n_iter))
freq_validation_loss = np.zeros((len(L_array), n_iter))

for i, L in enumerate(L_array):
    print('iteration {}'.format(L))
    print('compiling...')
    comparator = SequentialComparatorWithAlgotithmSelection(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                      n_validation_sample=validation_size, train_freq=True, train_bayes=False, train_shared_bayes=False)
    print('training...')
    for _ in range(n_iter):
        comparator.train_iteration()

    freq_train_loss[i] = comparator.freq_train_loss
    freq_validation_loss[i] = comparator.freq_validation_loss

    with open('number_of_layers_freq_{}.pkl'.format(L), 'wb') as f:
        pickle.dump(comparator, f)


plt.plot(L_array, freq_train_loss[:, -1], label="freq train loss")
plt.plot(L_array, freq_validation_loss[:, -1], label="freq valid loss")

plt.legend()
plt.show()

