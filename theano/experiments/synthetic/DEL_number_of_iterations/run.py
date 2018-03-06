from compare_freq_bayes.compare_sequential import SequentialComparator
import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

np.random.seed(10)

D = 10
K = 8
L = 4

n_iter_array = np.linspace(1000, 10000, 10)

batch_size = 1000
validation_size = 100

freq_train_loss = np.zeros(len(n_iter_array))
freq_validation_loss = np.zeros(len(n_iter_array))
freq_train_f_meas = np.zeros(len(n_iter_array))
freq_validation_f_meas = np.zeros(len(n_iter_array))
bayes_train_loss = np.zeros(len(n_iter_array))
bayes_validation_loss = np.zeros(len(n_iter_array))
bayes_train_f_meas = np.zeros(len(n_iter_array))
bayes_validation_f_meas = np.zeros(len(n_iter_array))

print('compiling...')
comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size,
                                      n_validation_sample=validation_size, train_freq=True, train_bayes=False, train_shared_bayes=False)
print('training...')

for _ in range(n_iter_array[-1]):
        comparator.train_iteration()

    freq_train_loss[i] = comparator.freq_train_loss[-1]
    freq_validation_loss[i] = comparator.freq_validation_loss[-1]
    freq_train_f_meas[i] = comparator.freq_train_loss[-1]
    freq_validation_f_meas[i] = comparator.freq_validation_loss[-1]
    bayes_train_loss[i] = comparator.freq_train_loss[-1]
    bayes_validation_loss[i] = comparator.freq_validation_loss[-1]
    bayes_train_f_meas[i] = comparator.freq_train_loss[-1]
    bayes_validation_f_meas[i] = comparator.freq_validation_loss[-1]

    with open('number_of_iterations_freq_{}.pkl'.format(L), 'wb') as f:
        pickle.dump(comparator, f)


plt.plot(n_iter_array, freq_train_loss, label="freq train loss")
plt.plot(n_iter_array, freq_validation_loss, label="freq valid loss")

plt.legend()
plt.show()

