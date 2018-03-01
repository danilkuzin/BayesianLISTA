import numpy as np
import theano
import six.moves.cPickle as pickle

from PBP_net_lista.DataGenerator import DataGenerator
from PBP_net_lista.net_lista import net_lista
from PBP_net_lista_single_matrices.net_lista import net_lista as net_list_single_matrices
from freqLISTA.run_lista import sgd_optimization_lista, predict

import matplotlib.pyplot as plt


def create_data(dataGenerator, n, borrow=True):
    beta, y, _ = dataGenerator.new_sample(n)
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
    shared_beta = theano.shared(np.asarray(beta, dtype=theano.config.floatX), borrow=borrow)
    return shared_beta, shared_y

def train_freq_lista(beta_train, y_train, D, K, L, X, n_train_batches, learning_rate, n_epochs, batch_size):
    lista = sgd_optimization_lista(beta_train=beta_train, y_train=y_train, X=X, n_train_batches=n_train_batches, learning_rate=learning_rate, n_epochs=n_epochs,
                           batch_size=batch_size, D=D, K=K, L=L)
    with open('best_model_freq_lista.pkl', 'wb') as f:
        pickle.dump(lista, f)

def predict_freq_lista(y_test):
    lista = pickle.load(open('best_model_freq_lista.pkl', 'rb'))

    beta_estimator = predict(lista, y_test=y_test)
    return beta_estimator

def train_bayesian_lista(beta_train, y_train, L, n_epochs):
    net = net_lista(Beta_train=beta_train, Y_train=y_train, L=L, n_epochs=n_epochs)
    net.sample_weights()

    with open('best_model_bayes_lista.pkl', 'wb') as f:
        pickle.dump(net, f)

def predict_bayesian_lista(y_test):
    net = pickle.load(open('best_model_bayes_lista.pkl', 'rb'))

    w, m, v, v_noise = net.predict(y_test)
    beta = net.predict_deterministic(y_test)
    return beta, w, m, v, v_noise

def train_bayesian_lista_single_matrices(beta_train, y_train, L, n_epochs):
    net = net_list_single_matrices(Beta_train=beta_train, Y_train=y_train, L=L, n_epochs=n_epochs)
    net.sample_weights()

    with open('best_model_bayes_lista_single_matrices.pkl', 'wb') as f:
        pickle.dump(net, f)

def predict_bayesian_lista_single_matrices(y_test):
    net = pickle.load(open('best_model_bayes_lista_single_matrices.pkl', 'rb'))

    w, m, v, v_noise = net.predict(y_test)
    beta = net.predict_deterministic(y_test)
    return beta, w, m, v, v_noise

if __name__ == '__main__':
    np.random.seed(1)

    D = 10
    K = 8
    L = 4

    dataGenerator = DataGenerator(D, K)

    n_train_batches = 10000
    batch_size = 100
    n_test_batches = 10

    # beta_train, y_train = create_data(dataGenerator, n_train_batches)


    beta_train, y_train, _ = dataGenerator.new_sample(n_train_batches)
    theano_y_train = theano.shared(np.asarray(y_train, dtype=theano.config.floatX))
    theano_beta_train = theano.shared(np.asarray(beta_train, dtype=theano.config.floatX))

    learning_rate = 0.0001
    n_epochs = 100

    #train_freq_lista(theano_beta_train, theano_y_train, D, K, L, dataGenerator.X, n_train_batches, learning_rate, n_epochs, batch_size)
    #
    # #this does not use batches so it has different training data size
    train_bayesian_lista(beta_train, y_train, L, n_epochs)

    train_bayesian_lista_single_matrices(beta_train, y_train, L, n_epochs)

    # beta_test, y_test = create_data(dataGenerator, n_train_batches)

    beta_test, y_test, _ = dataGenerator.new_sample(n_test_batches)
    theano_y_test = theano.shared(np.asarray(y_test, dtype=theano.config.floatX))
    theano_beta_test = theano.shared(np.asarray(beta_test, dtype=theano.config.floatX))

    #freq_beta_estimator = predict_freq_lista(theano_y_test)

    bayesian_beta_estimator, w, m, v, v_noise = predict_bayesian_lista(y_test)

    bayesian_beta_estimator_single_matrices, \
        w_single_matrices, m_single_matrices, \
        v_single_matrices, v_noise_single_matrices = predict_bayesian_lista_single_matrices(y_test)

    #print("beta_true:{}\n, predicted_lista:{}\n, predicted_pbp:{}\n, predicted_pbp_single_matrices:{}\n".format(beta_test, freq_beta_estimator, bayesian_beta_estimator, bayesian_beta_estimator_single_matrices))

    for i in range(beta_test.shape[0]):
        plt.plot(beta_test[i], label="true")
        #plt.plot(freq_beta_estimator[i], label="freq")
        plt.plot(bayesian_beta_estimator[i], label="bayesian")
        plt.plot(bayesian_beta_estimator_single_matrices[i], label="bayesian single")
        plt.legend()
        plt.show()

