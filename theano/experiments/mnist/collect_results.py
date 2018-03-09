import errno
import numpy as np
import os
import matplotlib.pyplot as plt

class MnistExperimentResultsCollector(object):
    def __init__(self):
        self.n_iter = 500

    def collect_all(self, undersampling):
        self.ista_f_meas_train, self.ista_f_meas_validation, self.ista_loss_train, self.ista_loss_validation = \
            self.collect_ista_fista('ista_fista/results_{}_K_ista.npz'.format(undersampling))
        self.fista_f_meas_train, self.fista_f_meas_validation, self.fista_loss_train, self.fista_loss_validation = \
            self.collect_ista_fista('ista_fista/results_{}_K_fista.npz'.format(undersampling))
        self.freq_f_meas_train, self.freq_f_meas_validation, self.freq_loss_train, self.freq_loss_validation, \
        self.sh_bayes_f_meas_train, self.sh_bayes_f_meas_validation, self.sh_bayes_loss_train, self.sh_bayes_loss_validation = \
            self.collect_main('normalised_results/mnist_100_train_20_layers_K_{}'.format(undersampling))

    def collect_ista_fista(self, fileprefix):
        saved_res = np.load(fileprefix)
        f_meas_train = saved_res['train_f_meas']
        f_meas_validation = saved_res['validation_f_meas']
        loss_train = saved_res['train_loss']
        loss_validation = saved_res['validation_loss']

        return f_meas_train, f_meas_validation, loss_train, loss_validation

    def collect_main(self, filename_prefix):
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

        return freq_train_f_measure, freq_validation_f_measure, freq_train_loss, freq_validation_loss, \
               shared_bayesian_train_f_measure, shared_bayesian_validation_f_measure, shared_bayesian_train_loss, shared_bayesian_validation_loss


        # return f_meas_train, f_meas_validation, loss_train, loss_validation, \
        #        f_meas_train_std, f_meas_validation_std, loss_train_std, loss_validation_std


    def plot_all(self):

        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = 1.
        markersize = 5.
        std_multiplier = 2

        ista_marker= None
        ista_color='blue'
        ista_label='ISTA'

        fista_marker= None
        fista_color='red'
        fista_label='FISTA'

        freq_marker= None
        freq_color='black'
        freq_label='freq LISTA'

        sh_bayes_marker=None
        sh_bayes_color='green'
        sh_bayes_label='bayes LISTA'

        plt.rc('text', usetex=True)

        plt.plot(self.ista_f_meas_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_f_meas_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_f_meas_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.sh_bayes_f_meas_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend()
        plt.ylabel(r'F measure')
        plt.xlabel(r'n iter')
        plt.savefig('results/normalised_f_measure_train.eps', format='eps')
        plt.show()

        plt.plot(self.ista_f_meas_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_f_meas_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_f_meas_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.sh_bayes_f_meas_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend()
        plt.ylabel(r'F measure')
        plt.xlabel(r'n iter')
        plt.savefig('results/normalised_f_measure_valid.eps', format='eps')
        plt.show()

        plt.plot(self.ista_loss_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_loss_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_loss_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.sh_bayes_loss_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend()
        plt.ylabel(r'NMSE')
        plt.xlabel(r'n iter')
        plt.savefig('results/normalised_nmse_train.eps', format='eps')
        plt.show()

        plt.plot(self.ista_loss_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_loss_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_loss_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.sh_bayes_loss_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend()
        plt.ylabel(r'NMSE')
        plt.xlabel(r'n iter')
        plt.savefig('results/normalised_nmse_valid.eps', format='eps')
        plt.show()


if __name__=='__main__':
    collector = MnistExperimentResultsCollector()
    collector.collect_all(250)
    collector.plot_all()