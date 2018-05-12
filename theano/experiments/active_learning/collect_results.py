import errno
import numpy as np
import os
import matplotlib.pyplot as plt

class ActiveExperimentMnistResultsCollector(object):
    def __init__(self):
        self.n_rseed = 20
        self.n_upd_iter = 10
        self.freq_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.non_active_bayes_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.active_bayes_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.freq_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.non_active_bayes_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.active_bayes_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))

    def collect_all(self):
        for rseed in range(self.n_rseed):
            res = np.load('mnist_results_normalised/mnist_active_rseed_{}.npz'.format(rseed))
            self.freq_validation_loss[rseed] = res['freq_validation_loss']
            self.non_active_bayes_validation_loss[rseed] = res['non_active_bayes_validation_loss']
            self.active_bayes_validation_loss[rseed] = res['active_bayes_validation_loss']
            self.freq_validation_f_measure[rseed] = res['freq_validation_f_measure']
            self.non_active_bayes_validation_f_measure[rseed] = res['non_active_bayes_validation_f_measure']
            self.active_bayes_validation_f_measure[rseed] = res['active_bayes_validation_f_measure']

    def plot_each(self):
        for rseed in range(self.n_rseed):
            res = np.load('mnist_results_normalised/mnist_active_rseed_{}.npz'.format(rseed))

            plt.plot(res['active_bayes_validation_loss'], label='active')
            plt.plot(res['non_active_bayes_validation_loss'], label='non active')
            plt.legend()
            plt.show()

    def plot_all(self):
        try:
            os.makedirs('mnist_results_plots')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = .25
        markersize = 5.
        std_multiplier = 1
        label_fontsize = 18
        legend_fontsize = 14

        non_active_bayes_marker = 'v'
        non_active_bayes_color = 'red'
        non_active_bayes_label = 'non active bayes'

        freq_marker = 's'
        freq_color = 'black'
        freq_label = 'freq lista'

        active_bayes_marker = 'd'
        active_bayes_color = 'green'
        active_bayes_label = 'active bayes'

        # plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_loss, axis=0),
        #          label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        # plt.errorbar(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_loss, axis=0),
        #              yerr=std_multiplier * np.std(self.freq_validation_loss, axis=0), color=freq_color)
        plt.plot(np.arange(0, self.n_upd_iter + 1), np.mean(self.active_bayes_validation_loss, axis=0),
                 label=active_bayes_label, color=active_bayes_color, marker=active_bayes_marker, linewidth=linewidth,
                 markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.active_bayes_validation_loss, axis=0),
                     yerr=std_multiplier * np.std(self.active_bayes_validation_loss, axis=0), color=active_bayes_color)
        plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.non_active_bayes_validation_loss, axis=0),
                 label=non_active_bayes_label, color=non_active_bayes_color, marker=non_active_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.non_active_bayes_validation_loss, axis=0),
                     yerr=std_multiplier * np.std(self.non_active_bayes_validation_loss, axis=0), color=non_active_bayes_color)

        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'NMSE', fontsize=label_fontsize)
        plt.savefig('mnist_results_plots/nmse_validation.eps', format='eps')
        plt.show()

        # plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_f_measure, axis=0),
        #          label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        # plt.errorbar(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_f_measure, axis=0),
        #              yerr=std_multiplier * np.std(self.freq_validation_f_measure, axis=0), color=freq_color)
        plt.plot(np.arange(0, self.n_upd_iter + 1), np.mean(self.active_bayes_validation_f_measure, axis=0),
                 label=active_bayes_label, color=active_bayes_color, marker=active_bayes_marker, linewidth=linewidth,
                 markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.active_bayes_validation_f_measure, axis=0),
                     yerr=std_multiplier * np.std(self.active_bayes_validation_f_measure, axis=0), color=active_bayes_color)
        plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.non_active_bayes_validation_f_measure, axis=0),
                 label=non_active_bayes_label, color=non_active_bayes_color, marker=non_active_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.non_active_bayes_validation_f_measure, axis=0),
                     yerr=std_multiplier * np.std(self.non_active_bayes_validation_f_measure, axis=0), color=non_active_bayes_color)

        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('mnist_results_plots/f_measure_validation.eps', format='eps')
        plt.show()

class ActiveExperimentSyntheticResultsCollector(object):
    def __init__(self):
        self.n_rseed = 10
        self.n_upd_iter = 10
        self.freq_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.non_active_bayes_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.active_bayes_validation_loss = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.freq_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.non_active_bayes_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))
        self.active_bayes_validation_f_measure = np.zeros((self.n_rseed, self.n_upd_iter+1))

    def collect_all(self):
        for rseed in range(self.n_rseed):
            res = np.load('synthetic_results/synthetic_active_rseed_{}.npz'.format(rseed))
            self.freq_validation_loss[rseed] = res['freq_validation_loss']
            self.non_active_bayes_validation_loss[rseed] = res['non_active_bayes_validation_loss']
            self.active_bayes_validation_loss[rseed] = res['active_bayes_validation_loss']
            self.freq_validation_f_measure[rseed] = res['freq_validation_f_measure']
            self.non_active_bayes_validation_f_measure[rseed] = res['non_active_bayes_validation_f_measure']
            self.active_bayes_validation_f_measure[rseed] = res['active_bayes_validation_f_measure']

    def plot_all(self):
        try:
            os.makedirs('synthetic_results_plots')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = .25
        markersize = 5.
        std_multiplier = 2
        label_fontsize = 24
        legend_fontsize = 14

        non_active_bayes_marker = 'v'
        non_active_bayes_color = 'red'
        non_active_bayes_label = 'non active bayes'

        freq_marker = 's'
        freq_color = 'black'
        freq_label = 'freq lista'

        active_bayes_marker = 'd'
        active_bayes_color = 'green'
        active_bayes_label = 'active bayes'

        plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_loss, axis=0),
                 label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter+1), np.mean(self.freq_validation_loss, axis=0),
                     yerr=std_multiplier * np.std(self.freq_validation_loss, axis=0), color=freq_color)
        plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.non_active_bayes_validation_loss, axis=0),
                 label=non_active_bayes_label, color=non_active_bayes_color, marker=non_active_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.non_active_bayes_validation_loss, axis=0),
                     yerr=std_multiplier * np.std(self.non_active_bayes_validation_loss, axis=0), color=non_active_bayes_color)
        plt.plot(np.arange(0, self.n_upd_iter+1), np.mean(self.active_bayes_validation_loss, axis=0),
                 label=active_bayes_label, color=active_bayes_color, marker=active_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(np.arange(0, self.n_upd_iter + 1), np.mean(self.active_bayes_validation_loss, axis=0),
                     yerr=std_multiplier * np.std(self.active_bayes_validation_loss, axis=0), color=active_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'NMSE', fontsize=label_fontsize)
        plt.savefig('synthetic_results_plots/nmse_validation.eps', format='eps')
        plt.show()

if __name__=='__main__':
    collector = ActiveExperimentMnistResultsCollector()
    collector.collect_all()
    collector.plot_all()