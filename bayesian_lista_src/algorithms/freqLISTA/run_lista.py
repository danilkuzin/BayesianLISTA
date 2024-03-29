import timeit

from PBP_net_lista.DataGenerator import DataGenerator
from freqLISTA.lista import Lista
import numpy as np
import theano.tensor as T
import theano
import six.moves.cPickle as pickle

import os
import sys

import matplotlib.pyplot as plt


def load_data(n_train_batches, n_valid_batches, n_test_batches, D, K, data_generator):
    def create_data(dataGenerator, n, borrow=True):
        beta, y, _ = dataGenerator.new_sample(n)
        shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        shared_beta = theano.shared(np.asarray(beta, dtype=theano.config.floatX), borrow=borrow)
        return shared_beta, shared_y

    beta_train, y_train = create_data(dataGenerator, n_train_batches)
    beta_test, y_test = create_data(dataGenerator, n_valid_batches)
    beta_valid, y_valid = create_data(dataGenerator, n_test_batches)

    return [(beta_train, y_train), (beta_valid, y_valid), (beta_test, y_test)]


def sgd_optimization_lista(beta_train, y_train, X, n_train_batches=10, learning_rate=0.13, n_epochs=1000,
                           batch_size=600, D=10, K=5, L=6):
    print('... building the model')
    index = T.lscalar()

    y = T.matrix('y')
    beta = T.matrix('beta')

    lista = Lista(L, D, K, y, X)

    cost = lista.mean_squared_error(beta)

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

    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)

    end_time = timeit.default_timer()
    print('Optimization complete, {} sec'.format(end_time - start_time))
    return lista


def sgd_optimization_lista_old(n_train_batches=10, n_valid_batches=10, n_test_batches=10, learning_rate=0.13,
                                        n_epochs=1000,
                                        batch_size=600, D=10, K=5, L=6, data_generator=None):
    datasets = load_data(n_train_batches * batch_size, n_valid_batches, n_test_batches, D=D, K=K,
                         data_generator=data_generator)
    beta_train, y_train = datasets[0]
    beta_valid, y_valid = datasets[1]
    beta_test, y_test = datasets[2]

    print('... building the model')
    index = T.lscalar()

    y = T.matrix('y')
    beta = T.matrix('beta')

    lista = Lista(L, D, K, y, data_generator.X)

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

            print(
                'epoch %i, minibatch %i/%i, train error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    minibatch_avg_cost
                )
            )

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


def predict(lista, y_test):
    predict_model = theano.function(
        inputs=[lista.y],
        outputs=lista.beta_estimator
    )
    y_test = y_test.get_value()

    predicted_values = predict_model(y_test)

    return predicted_values


def predict_old(D, K, data_generator):
    lista = pickle.load(open('best_model.pkl', 'rb'))

    datasets = load_data(n_train_batches=10, n_valid_batches=10, n_test_batches=10, D=D, K=K,
                         data_generator=data_generator)
    beta_test, y_test = datasets[2]

    predict_model = theano.function(
        inputs=[lista.y],
        outputs=lista.beta_estimator
    )
    y_test = y_test.get_value()
    beta_test = beta_test.get_value()

    test_number = 3

    predicted_values = predict_model(y_test[:test_number])
    print("Predicted values for the first 1 examples in test set:")
    print(predicted_values)
    print("True values:")
    print(beta_test[:test_number])

    for t in range(test_number):
        plt.plot(predicted_values[t], label='predicted')
        plt.plot(beta_test[t], label='true')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'None'

    np.random.seed(1)
    D = 10
    K = 9
    dataGenerator = DataGenerator(D, K)
    sgd_optimization_lista_old(n_train_batches=1000, n_valid_batches=20, n_test_batches=20, learning_rate=0.00001,
                           n_epochs=10000,
                           batch_size=100, D=D, K=K, L=2, data_generator=dataGenerator)
    predict_old(D=D, K=K, data_generator=dataGenerator)
