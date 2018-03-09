import numpy as np
from PBP_net_lista.test_network_layer import random_spike_and_slab
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from PBP_net_lista_single_matrices.SingleBayesianListaHandler import SingleBayesianListaHandler


class PosteriorPlotter(object):

    def plot_posterior_samples_mnist(self, bayes_lista, y, n_samples):
        w, m, v = bayes_lista.pbp_instance.predict_probabilistic(y)
        for sample_num in range(n_samples):
            beta_est_sample = random_spike_and_slab(m, np.sqrt(v), w)
            plt.figure()
            plt.imshow(np.reshape(beta_est_sample, (28, 28)), cmap='gray')
            plt.axis('off')
            plt.savefig('posterior_sample_{}.pdf'.format(sample_num), format='pdf', bbox_inches='tight')
            plt.show()

    def plot_posterior_parameters_mnist(self, bayes_lista, y):
        w, m, v = bayes_lista.pbp_instance.predict_probabilistic(y)

        plt.figure()
        plt.imshow(np.reshape(m, (28, 28)))
        plt.colorbar()
        plt.axis('off')
        plt.savefig('posterior_mean.pdf', format='pdf', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.imshow(np.reshape(np.sqrt(v), (28, 28)))
        plt.colorbar()
        plt.axis('off')
        plt.savefig('posterior_std.pdf', format='pdf', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.imshow(np.reshape(w, (28, 28)))
        plt.colorbar()
        plt.axis('off')
        plt.savefig('posterior_spike_indicator.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def plot_loss(self, comparator):
        plt.figure()
        plt.semilogy(comparator.freq_train_loss, label="freq train loss")
        #plt.semilogy(comparator.bayesian_train_loss, label="bayes train loss")
        plt.semilogy(comparator.shared_bayesian_train_loss, label="shared bayes train loss")

        plt.semilogy(comparator.freq_validation_loss, label="freq valid loss")
        #plt.semilogy(comparator.bayesian_validation_loss, label="bayes valid loss")
        plt.semilogy(comparator.shared_bayesian_validation_loss, label="shared bayes valid loss")

        plt.legend()
        plt.savefig('loss.pdf', format='pdf', bbox_inches='tight')
        plt.show()



if __name__ == '__main__':

    np.random.seed(1)

    bayes_weights = np.load('./normalised_results/mnist_100_train_20_layers_K_250_bayes_weights.npz')
    dictionary = np.load('./normalised_results/mnist_100_train_20_layers_K_250_dictionary.npz')
    data = np.load('./normalised_results/mnist_100_train_20_layers_K_250_beta_est.npz')

    y_validation = data['y_validation']
    beta_validation = data['true_beta_validation']

    bayes_lista = SingleBayesianListaHandler(bayes_weights['D'],
                                             bayes_weights['K'], bayes_weights['L'], dictionary['X'])

    bayes_lista.pbp_instance.network.params_W_M.set_value(bayes_weights['bayes_W_M'])
    bayes_lista.pbp_instance.network.params_W_V.set_value(bayes_weights['bayes_W_V'])
    bayes_lista.pbp_instance.network.params_S_M.set_value(bayes_weights['bayes_S_M'])
    bayes_lista.pbp_instance.network.params_S_V.set_value(bayes_weights['bayes_S_V'])


    plotter = PosteriorPlotter()

    example_num = 0

    plotter.plot_posterior_samples_mnist(bayes_lista, y_validation[example_num], 3)
    plotter.plot_posterior_parameters_mnist(bayes_lista, y_validation[example_num])
