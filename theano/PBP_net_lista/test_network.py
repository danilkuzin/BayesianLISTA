import numpy as np

import network
import theano

L = 5

W_M = []
W_V = []
S_M = []
S_V = []

D = 10
K = 5

a_w = 6.0 * np.ones(D)
b_w = 6.0 * np.ones(D)

theano.config.exception_verbosity='high'
theano.config.optimizer='None'

for i in range(L):
    W_M.append(np.random.randn(D, K))
    W_V.append(np.random.rand(D, K))
    S_M.append(np.random.randn(D, D))
    S_V.append(np.random.rand(D, D))

thr_lambda = 0.1


network = network.Network(W_M, W_V, S_M, S_V, thr_lambda, a_w, b_w, D, K)

y = np.random.randn(K)
print('t')
for t in network.output_probabilistic(y):
    print(t.eval())

beta = np.random.randn(D)
print('z')
res_z = network.Z_Z1_Z2(beta, y)
for z in res_z:
    print(z.eval())

Z = res_z[0]
Z1 = res_z[1]
Z2 = res_z[2]
updates = network.generate_updates(Z, Z1, Z2)
print('u')
print(updates)