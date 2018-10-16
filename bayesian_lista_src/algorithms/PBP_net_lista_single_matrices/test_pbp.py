import pbp
import numpy as np

import theano

from PBP_net_lista.DataGenerator import DataGenerator

L = 2
D = 4
K = 3
mean_y_train = 1.0
std_y_train = 2.0

theano.config.exception_verbosity='high'
theano.config.optimizer='None'

Beta = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(pbp.network.get_params())
pbp.do_first_pass(Beta, Y)
print(pbp.network.get_params())
pbp.do_pbp(Beta, Y, 5)
print(pbp.network.get_params())