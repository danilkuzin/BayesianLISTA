from PBP_net_lista.net_lista import net_lista
import numpy as np

import theano

from PBP_net_lista.DataGenerator import DataGenerator

D = 10
K = 5
N = 10

# theano.config.exception_verbosity='high'
# theano.config.optimizer='None'

dataGenerator = DataGenerator(D, K)
Beta_train, Y_train, _ = dataGenerator.new_sample(N)
Beta_test, Y_test, _ = dataGenerator.new_sample(N)

L = 1

net = net_lista(Beta_train, Y_train, L)
w, m, v, v_noise = net.predict(Y_test)
print(w, m, v, v_noise)