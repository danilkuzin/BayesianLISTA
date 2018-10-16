import numpy as np

import prior
import theano

L = 2
D = 10
K = 5
var_targets = 3.0

pr = prior.Prior(L, D, K, var_targets)
print(pr)

params = pr.get_initial_params()
print(params)