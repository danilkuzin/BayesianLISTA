import timeit

from PBP_net_lista.DataGenerator import DataGenerator
from freqLISTA.lista import Lista
import numpy as np
import theano.tensor as T
import theano
import six.moves.cPickle as pickle

import os
import sys


def load_data(n_train_batches, n_valid_batches, n_test_batches, D, K):
    def create_data(dataGenerator, n, borrow=True):
        beta, y, _ = dataGenerator.new_sample(n)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        shared_beta = theano.shared(np.asarray(beta, dtype=theano.config.floatX), borrow=borrow)
        return shared_beta, shared_y

    dataGenerator = DataGenerator(D, K)
    beta_train, y_train = create_data(dataGenerator, n_train_batches)
    beta_test, y_test = create_data(dataGenerator, n_valid_batches)
    beta_valid, y_valid = create_data(dataGenerator, n_test_batches)

    return [(beta_train, y_train), (beta_valid, y_valid), (beta_test, y_test)]

def sgd_optimization_lista(n_train_batches=10, n_valid_batches=10, n_test_batches=10, learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz', batch_size=600, D=10, K=5, L=6):
    datasets = load_data(n_train_batches, n_valid_batches, n_test_batches, D=D, K=K)
    beta_train, y_train = datasets[0]
    beta_valid, y_valid = datasets[1]
    beta_test, y_test = datasets[2]

    print('... building the model')
    index = T.lscalar()

    y = T.matrix('y')
    beta = T.matrix('beta')

    # D = 10
    # K = 5
    # L = 6

    # batch_size = 50
    #
    # N_train = 10000
    # N_test = 1

    lista = Lista(L, D, K, y)

    cost = lista.mean_squared_error(beta)

    test_model = theano.function(
        inputs=[index],
        outputs=lista.errors(beta),
        givens={
            y: y_test[index * batch_size: (index + 1) * batch_size],
            beta: beta_test[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=lista.errors(beta),
        givens={
            y: y_valid[index * batch_size: (index + 1) * batch_size],
            beta: beta_valid[index * batch_size: (index + 1) * batch_size]
        }
    )

    learning_rate = 0.1

    g_S = T.grad(cost=cost, wrt=lista.S)
    g_W = T.grad(cost=cost, wrt=lista.W)

    updates = [(lista.S, lista.S - learning_rate * g_S),
               (lista.W, lista.W - learning_rate * g_W)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            y: y_train[index * batch_size: (index + 1) * batch_size],
            beta: beta_train[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training the model')
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(lista, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict():

    lista = pickle.load(open('best_model.pkl'))

    predict_model = theano.function(
        inputs=[lista.input],
        outputs=lista.beta_estimator)

    datasets = load_data(n_train_batches=10, n_valid_batches=10, n_test_batches=10, D=10, K=5)
    beta_train, y_train = datasets[0]
    beta_valid, y_valid = datasets[1]
    beta_test, y_test = datasets[2]

    predicted_values = predict_model(y_test[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)


if __name__ == '__main__':
    np.random.seed(1)
    sgd_optimization_lista()