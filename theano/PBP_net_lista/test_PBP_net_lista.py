import net_lista
import numpy as np

from PBP_net_lista.DataGenerator import DataGenerator

D = 10
K = 5
N = 2

dataGenerator = DataGenerator(D, K)
Beta_train, Y_train, Noise = dataGenerator.new_sample(N)

L = 1

net = net_lista.net_lista(Beta_train, Y_train, L)