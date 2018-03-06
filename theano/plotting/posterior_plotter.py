import numpy as np
from PBP_net_lista.test_network_layer import random_spike_and_slab
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle

from compare_mnist.compare_mnist_script import MnistSequentialComparator
from compare_freq_bayes.compare_sequential import SequentialComparator


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

    saved_comparator_file_name = '../compare_freq_bayes/best_model_bayes_lista_single_matrices.pkl'
    comparator = pickle.load(open(saved_comparator_file_name, 'rb'))

    plotter = PosteriorPlotter()

    example_num = 7

    #plotter.plot_posterior_samples_mnist(comparator.bayesian_lista, comparator.data.y_validation[example_num], 3)
    #plotter.plot_posterior_parameters_mnist(comparator.bayesian_lista, comparator.data.y_validation[example_num])
    plotter.plot_loss(comparator)