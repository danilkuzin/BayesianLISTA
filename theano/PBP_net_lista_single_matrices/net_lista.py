
import numpy as np

import pickle

import gzip

from PBP_net_lista_single_matrices import pbp


class net_lista:

    def __init__(self, Beta_train, Y_train, L, n_epochs = 40,
        normalize = False):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_Beta_train = np.std(Beta_train, 0)
            self.std_Beta_train[ self.std_Beta_train == 0 ] = 1
            self.mean_Beta_train = np.mean(Beta_train, 0)
        else:
            self.std_Beta_train = np.ones(Beta_train.shape[ 1 ])
            self.mean_Beta_train = np.zeros(Beta_train.shape[ 1 ])

        Beta_train = (Beta_train - np.full(Beta_train.shape, self.mean_Beta_train)) / \
            np.full(Beta_train.shape, self.std_Beta_train)

        self.mean_Y_train = np.mean(Y_train, axis=0)
        self.std_Y_train = np.std(Y_train, axis=0)

        Y_train_normalized = (Y_train - self.mean_Y_train) / self.std_Y_train

        # We construct the network

        # n_units_per_layer = \
        #     np.concatenate(([ X_train.shape[ 1 ] ], n_hidden, [ 1 ]))

        N, D = Beta_train.shape
        N, K = Y_train.shape
        self.pbp_instance = \
            pbp.PBP_lista(L, D, K, self.mean_Y_train, self.std_Y_train)

        # We iterate the learning process

        self.pbp_instance.do_pbp(Beta_train, Y_train_normalized, n_epochs)

        # We are done!

    def re_train(self, X_train, y_train, n_epochs):

        """
            Function that re-trains the network on some data.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Numer of epochs for which to train the
                                network. 
        """

        # We normalize the training data 

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
            np.full(X_train.shape, self.std_X_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

    def predict(self, Y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        Y_test = np.array(Y_test, ndmin = 2)

        # We normalize the test set

        Y_test = (Y_test - np.full(Y_test.shape, self.mean_Y_train)) / \
            np.full(Y_test.shape, self.std_Y_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        w, m, v, v_noise = self.pbp_instance.get_predictive_omega_mean_and_variance(Y_test)

        # We are done!

        return w, m, v, v_noise

    def predict_deterministic(self, Y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param Y_test   The matrix of features for the test data
            
    
            @return o       The predictive value for the test target variables.

        """

        Y_test = np.array(Y_test, ndmin=2)

        # We normalize the test set

        Y_test = (Y_test - np.full(Y_test.shape, self.mean_Y_train)) / \
            np.full(Y_test.shape, self.std_Y_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        o = self.pbp_instance.get_deterministic_output(Y_test)

        # We are done!

        return o

    def sample_weights(self):

        """
            Function that draws a sample from the posterior approximation
            to the weights distribution.

        """
 
        self.pbp_instance.sample_ws()

    def save_to_file(self, filename):

        """
            Function that stores the network in a file.

            @param filename   The name of the file.
            
        """

        # We save the network to a file using pickle

        def save_object(obj, filename):

            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
            dest.close()

        save_object(self, filename)

def load_PBP_net_from_file(filename):

    """
        Function that load a network from a file.

        @param filename   The name of the file.
        
    """

    def load_object(filename):

        with gzip.GzipFile(filename, 'rb') as \
            source: result = source.read()
        ret = pickle.loads(result)
        source.close()

        return ret

    # We load the dictionary with the network parameters

    PBP_network = load_object(filename)

    return PBP_network
