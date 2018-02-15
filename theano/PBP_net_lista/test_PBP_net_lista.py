from PBP_net_lista.net_lista import net_lista
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.test_network_layer import random_spike_and_slab

np.random.seed(1)

D = 10
K = 5
N_train = 10000
N_test = 1

# theano.config.exception_verbosity='high'
# theano.config.optimizer='None'

dataGenerator = DataGenerator(D, K)
Beta_train, Y_train, _ = dataGenerator.new_sample(N_train)
Beta_test, Y_test, _ = dataGenerator.new_sample(N_test)

L = 5

net = net_lista(Beta_train, Y_train, L)
w, m, v, v_noise = net.predict(Y_test)
net.sample_weights()
beta = net.predict_deterministic(Y_test)
print("w:{}\n, m:{}\n, v:{}\n, v_noise:{}\n, beta: {}\n".format(w, m, v, v_noise, beta))

n = 0

sample_size = 100000

sample_z = np.zeros((sample_size, D), dtype=np.float64)
for i in range(sample_size):
    sample_z[i, :] = random_spike_and_slab(m[n], np.sqrt(v[n]), w[n])

for d in range(D):
    print("Sample {0}. Parameters for dim{1}. w:{2}, m:{3}, v:{4}, beta:{5}\n".format(n, d, w[n, d], m[n, d], v[n, d], Beta_test[n, d]))
    sns.distplot(sample_z[:, d])
    plt.plot((Beta_test[n, d], Beta_test[n, d]), (0, 1), 'k-')
    plt.show()

