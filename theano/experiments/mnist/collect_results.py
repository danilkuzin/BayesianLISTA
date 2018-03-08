import errno
import numpy as np
import os
import matplotlib.pyplot as plt

class MnistExperimentResultsCollector(object):
    def __init__(self):
        self.n_iter = 500

    def collect_all(self):
        self.ista_f_meas_train, self.ista_f_meas_validation, self.ista_loss_train, self.ista_loss_validation, \
        self.ista_f_meas_train_std, self.ista_f_meas_validation_std, self.ista_loss_train_std, self.ista_loss_validation_std = \
            self.collect_ista_fista('ista')
        self.fista_f_meas_train, self.fista_f_meas_validation, self.fista_loss_train, self.fista_loss_validation, \
        self.fista_f_meas_train_std, self.fista_f_meas_validation_std, self.fista_loss_train_std, self.fista_loss_validation_std, = \
            self.collect_ista_fista('fista')
        self.freq_f_meas_train, self.freq_f_meas_validation, self.freq_loss_train, self.freq_loss_validation, \
        self.freq_f_meas_train_std, self.freq_f_meas_validation_std, self.freq_loss_train_std, self.freq_loss_validation_std = \
            self.collect_main('freq')
        self.sh_bayes_f_meas_train, self.sh_bayes_f_meas_validation, self.sh_bayes_loss_train, self.sh_bayes_loss_validation,\
        self.sh_bayes_f_meas_train_std, self.sh_bayes_f_meas_validation_std, self.sh_bayes_loss_train_std, self.sh_bayes_loss_validation_std = \
            self.collect_main('sh_bayes')

    def collect(self, filename_prefix):
        saved_bayes_weights = np.load('{}_bayes_weights.npz'.format(filename_prefix))
        bayes_S_M = saved_bayes_weights['bayes_S_M']
        bayes_S_V = saved_bayes_weights['bayes_S_V']
        bayes_W_M = saved_bayes_weights['bayes_W_M']
        bayes_W_V = saved_bayes_weights['bayes_W_V']

        saved_beta_est = np.load('{}_beta_est.npz'.format(filename_prefix))
        freq_beta_train_est=saved_beta_est['freq_beta_train_est']
        freq_beta_validation_est=saved_beta_est['freq_beta_validation_est']
        shared_beta_train_est=saved_beta_est['shared_beta_train_est']
        shared_beta_validation_est=saved_beta_est['shared_beta_validation_est']
        true_beta_train=saved_beta_est['true_beta_train']
        true_beta_validation=saved_beta_est['true_beta_validation']
        y_train=saved_beta_est['y_train']
        y_validation=saved_beta_est['y_validation']

        saved_quality = np.load('{}_quality.npz'.format(filename_prefix))
        freq_train_f_measure=saved_quality['freq_train_f_measure']
        freq_train_loss=saved_quality['freq_train_loss']
        freq_validation_f_measure=saved_quality['freq_validation_f_measure']
        freq_validation_loss=saved_quality['freq_validation_loss']
        shared_bayesian_train_f_measure=saved_quality['shared_bayesian_train_f_measure']
        shared_bayesian_train_loss=saved_quality['shared_bayesian_train_loss']
        shared_bayesian_validation_f_measure=saved_quality['shared_bayesian_validation_f_measure']
        shared_bayesian_validation_loss=saved_quality['shared_bayesian_validation_loss']

        saved_params = np.load('{}_params.npz'.format(filename_prefix))
        D=saved_params['D']
        K=saved_params['K']
        L=saved_params['L']
        train_size=saved_params['train_size']

        plt.plot(freq_train_f_measure, label='freq train f meas')
        plt.plot(freq_validation_f_measure, label='freq valid f meas')
        plt.plot(shared_bayesian_train_f_measure, label='sh bayes train f meas')
        plt.plot(shared_bayesian_validation_f_measure, label='sh bayes train f meas')

        plt.legend()
        plt.show()

        plt.plot(freq_train_loss, label='freq train loss')
        plt.plot(freq_validation_loss, label='freq valid loss')
        plt.plot(shared_bayesian_train_loss, label='sh bayes train loss')
        plt.plot(shared_bayesian_validation_loss, label='sh bayes train loss')

        plt.legend()
        plt.show()

        # return f_meas_train, f_meas_validation, loss_train, loss_validation, \
        #        f_meas_train_std, f_meas_validation_std, loss_train_std, loss_validation_std


    def plot_all(self):

        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = .25
        markersize = 5.
        std_multiplier = 2

        ista_marker= 'o'
        ista_color='blue'
        ista_label='ista'

        fista_marker= 'v'
        fista_color='red'
        fista_label='fista'

        freq_marker= 's'
        freq_color='black'
        freq_label='freq lista'

        sh_bayes_marker='d'
        sh_bayes_color='green'
        sh_bayes_label='sh bayes'

        plt.plot(self.L_array, self.ista_f_meas_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.ista_f_meas_train, yerr=std_multiplier * self.ista_f_meas_train_std, color=ista_color)
        plt.plot(self.L_array, self.fista_f_meas_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.fista_f_meas_train, yerr=std_multiplier * self.fista_f_meas_train_std, color=fista_color)
        plt.plot(self.L_array, self.freq_f_meas_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.freq_f_meas_train, yerr=2 * self.freq_f_meas_train_std, color=freq_color)
        plt.plot(self.L_array, self.sh_bayes_f_meas_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.sh_bayes_f_meas_train, yerr=2 * self.sh_bayes_f_meas_train_std, color=sh_bayes_color)
        plt.legend()
        plt.savefig('results/f_measure_train.eps', format='eps')
        plt.xlabel(r'K')
        plt.ylabel(r'F measure')
        plt.show()

        plt.plot(self.L_array, self.ista_f_meas_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.ista_f_meas_validation, yerr=3 * self.ista_f_meas_validation_std, color=ista_color)
        plt.plot(self.L_array, self.fista_f_meas_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.fista_f_meas_validation, yerr=3 * self.fista_f_meas_validation_std, color=fista_color)
        plt.plot(self.L_array, self.freq_f_meas_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.freq_f_meas_validation, yerr=3 * self.freq_f_meas_validation_std, color=freq_color)
        plt.plot(self.L_array, self.sh_bayes_f_meas_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.sh_bayes_f_meas_validation, yerr=3 * self.sh_bayes_f_meas_validation_std, color=sh_bayes_color)
        plt.legend()
        plt.savefig('results/f_measure_validation.eps', format='eps')
        plt.xlabel(r'K')
        plt.ylabel(r'F measure')
        plt.show()

        plt.plot(self.L_array, self.ista_loss_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.ista_loss_train, yerr=3 * self.ista_loss_train_std, color=ista_color)
        plt.plot(self.L_array, self.fista_loss_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.fista_loss_train, yerr=3 * self.fista_loss_train_std, color=fista_color)
        plt.plot(self.L_array, self.freq_loss_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.freq_loss_train, yerr=3 * self.freq_loss_train_std, color=freq_color)
        plt.plot(self.L_array, self.sh_bayes_loss_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.sh_bayes_loss_train, yerr=3 * self.sh_bayes_loss_train_std, color=sh_bayes_color)
        plt.legend()
        plt.savefig('results/nmse_train.eps', format='eps')
        plt.xlabel(r'K')
        plt.ylabel(r'NMSE')
        plt.show()

        plt.plot(self.L_array, self.ista_loss_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.ista_loss_validation, yerr=3 * self.ista_loss_validation_std, color=ista_color)
        plt.plot(self.L_array, self.fista_loss_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.fista_loss_validation, yerr=3 * self.fista_loss_validation_std, color=fista_color)
        plt.plot(self.L_array, self.freq_loss_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.freq_loss_validation, yerr=3 * self.freq_loss_validation_std, color=freq_color)
        plt.plot(self.L_array, self.sh_bayes_loss_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.L_array, self.sh_bayes_loss_validation, yerr=3 * self.sh_bayes_loss_validation_std, color=sh_bayes_color)
        plt.legend()
        plt.savefig('results/nmse_validation.eps', format='eps')
        plt.xlabel(r'K')
        plt.ylabel(r'NMSE')
        plt.show()


if __name__=='__main__':
    collector = MnistExperimentResultsCollector()
    collector.collect('non_normalised_results/mnist_100_train_20_layers_K_250')
    # collector.collect_all()
    # collector.plot_all()