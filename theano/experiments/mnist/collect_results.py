import errno
import numpy as np
import os
import matplotlib.pyplot as plt

class MnistExperimentResultsCollector(object):
    def __init__(self):
        self.n_iter = 500

    def collect_main(self, filename_prefix):
        res = np.load('{}.npz'.format(filename_prefix))
        self.freq_time = res['lista_time']
        self.shared_bayesian_time = res['sh_bayes_time']
        self.freq_validation_loss = res['lista_nmse']
        self.freq_validation_f_measure = res['lista_f_meas']
        self.shared_bayesian_validation_loss = res['sh_bayes_nmse']
        self.shared_bayesian_validation_f_measure = res['sh_bayes_f_meas']
        self.ista_time = res['ista_time']
        self.fista_time = res['fista_time']
        self.ista_validation_loss = res['ista_nmse']
        self.fista_validation_loss = res['fista_nmse']
        self.ista_validation_f_measure = res['ista_f_meas']
        self.fista_validation_f_measure = res['fista_f_meas']

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

        label_fontsize = 24
        legend_fontsize = 14

        plt.rc('text', usetex=True)

        plt.plot(self.ista_validation_f_measure, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_validation_f_measure, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_validation_f_measure, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.shared_bayesian_validation_f_measure, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend(fontsize=legend_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.xlabel(r'Number of iterations', fontsize=label_fontsize)
        plt.savefig('results/f_measure_valid.eps', format='eps')
        plt.show()

        plt.plot(self.ista_validation_loss, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.fista_validation_loss, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_validation_loss, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.shared_bayesian_validation_loss, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend(fontsize=legend_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.xlabel(r'Number of iterations', fontsize=label_fontsize)
        plt.savefig('results/nmse_valid.eps', format='eps')
        plt.show()

        # plt.plot(self.ista_time, self.ista_validation_f_measure, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        # plt.plot(self.fista_time, self.fista_validation_f_measure, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_time, self.freq_validation_f_measure, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.shared_bayesian_time, self.shared_bayesian_validation_f_measure, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend(fontsize=legend_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.xlabel(r'Time, s', fontsize=label_fontsize)
        plt.xlim([0, 2500])
        plt.savefig('results/f_measure_valid_time.eps', format='eps')
        plt.show()

        # plt.plot(self.ista_time, self.ista_validation_loss, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        # plt.plot(self.fista_time, self.fista_validation_loss, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.freq_time, self.freq_validation_loss, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.shared_bayesian_time, self.shared_bayesian_validation_loss, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.plot()
        plt.legend(fontsize=legend_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.xlabel(r'Time, s', fontsize=label_fontsize)
        plt.xlim([0, 2500])
        plt.savefig('results/nmse_valid_time.eps', format='eps')
        plt.show()


if __name__=='__main__':
    collector = MnistExperimentResultsCollector()
    collector.collect_main('time_100_train_20_layers_K_250')
    collector.plot_all()