import time

from PBP_net_lista.network_layer import theano_soft_threshold

import theano.tensor as T
import numpy as np
import theano

patience = 5000
patience_increase = 2
n_train_batches = 1000
n_epochs = 100

improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience/2)

best_params = None
best_validation_loss = np.inf
test_score = 0.
start_time = time.clock()

D = 10
K = 5
L = 6

done_looping = False
epoch = 0

batch_size = 50

beta_estimator = T.vector()
beta_true = T.vector()
y = T.vector()

W = theano.shared(value=np.random.randn(D, K), name='W', borrow=True)
S = theano.shared(value=np.random.randn(D, D), name='S', borrow=True)
thr_lambda = theano.shared(value=0.1, name='thr_lambda', borrow=True)
params = [W, S]

learning_rate = 0.1

B = T.dot(W, y)
beta_estimator_history = [theano_soft_threshold(B, thr_lambda)]
for l in range(1, L):
    C = B + T.dot(S, beta_estimator_history[-1])
    beta_estimator_history.append(theano_soft_threshold(C, thr_lambda))
beta_estimator = beta_estimator_history[-1]

# loss_function = theano.function([beta_estimator, beta_true],
#                        1/batch_size * T.sum(0.5 * T.sum(T.sqr(beta_estimator - beta_true))))
loss_function = theano.function([beta_estimator, beta_true],
                       1/batch_size * T.sum(0.5 * T.sum(T.sqr(beta_estimator - beta_true))))

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):

        loss = loss_function(beta_estimator_train, beta_true_train)
        d_loss_wrt_params = T.grad(loss, [S, W])
        params -= learning_rate * d_loss_wrt_params

        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:

            this_validation_loss = loss(beta_estimator_validation, beta_true_validation)

            if this_validation_loss < best_validation_loss:

                if this_validation_loss < best_validation_loss * improvement_threshold:

                    patience = max(patience, iter * patience_increase)
                best_params = copy.deepcopy(params)
                best_validation_loss = this_validation_loss

        if patience <= iter:
            done_looping = True
            break


def net(self):

