import errno
import numpy as np
import os
import matplotlib.pyplot as plt

class UndersamplingExperimentReultsCollector(object):
    def __init__(self):
        self.K_array = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.random_range = 10
        self.n_iter = 1000

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

    def collect_ista_fista(self, filename):
        f_meas_train = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        f_meas_validation = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        loss_train = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        loss_validation = np.zeros((self.random_range, len(self.K_array), self.n_iter))

        for rseed in range(self.random_range):
            for i, K in enumerate(self.K_array):
                saved_res = np.load('ista_fista/fista_ista_{}_K_{}_rseed_{}.npz'.format(K, rseed, filename))
                f_meas_train[rseed, i] = saved_res['train_f_meas']
                f_meas_validation[rseed, i] = saved_res['validation_f_meas']
                loss_train[rseed, i] = saved_res['train_loss']
                loss_validation[rseed, i] = saved_res['validation_loss']

        f_meas_train_std = np.std(f_meas_train, axis=0)
        f_meas_validation_std = np.std(f_meas_train, axis=0)
        loss_train_std = np.std(f_meas_train, axis=0)
        loss_validation_std = np.std(f_meas_train, axis=0)

        f_meas_train_std = f_meas_train_std[:, -1]
        f_meas_validation_std = f_meas_validation_std[:, -1]
        loss_train_std = loss_train_std[:, -1]
        loss_validation_std = loss_validation_std[:, -1]

        f_meas_train = np.mean(f_meas_train, axis=0)
        f_meas_validation = np.mean(f_meas_validation, axis=0)
        loss_train = np.mean(loss_train, axis=0)
        loss_validation = np.mean(loss_validation, axis=0)

        f_meas_train = f_meas_train[:, -1]
        f_meas_validation = f_meas_validation[:, -1]
        loss_train = loss_train[:, -1]
        loss_validation = loss_validation[:, -1]


        return f_meas_train, f_meas_validation, loss_train, loss_validation, \
               f_meas_train_std, f_meas_validation_std, loss_train_std, loss_validation_std

    def collect_main(self, alg_name):
        f_meas_train = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        f_meas_validation = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        loss_train = np.zeros((self.random_range, len(self.K_array), self.n_iter))
        loss_validation = np.zeros((self.random_range, len(self.K_array), self.n_iter))

        for i in range(self.random_range):
            saved_res = np.load('{}/undersampling_measures.npz'.format(i+1, i+1))
            loss_train[i] = saved_res['{}_train_loss'.format(alg_name)]
            loss_validation[i] = saved_res['{}_validation_loss'.format(alg_name)]
            f_meas_train[i] = saved_res['{}_train_f_measure'.format(alg_name)]
            f_meas_validation[i] = saved_res['{}_validation_f_measure'.format(alg_name)]

        f_meas_train_std = np.std(f_meas_train, axis=0)
        f_meas_validation_std = np.std(f_meas_train, axis=0)
        loss_train_std = np.std(f_meas_train, axis=0)
        loss_validation_std = np.std(f_meas_train, axis=0)

        f_meas_train_std = f_meas_train_std[:, -1]
        f_meas_validation_std = f_meas_validation_std[:, -1]
        loss_train_std = loss_train_std[:, -1]
        loss_validation_std = loss_validation_std[:, -1]

        f_meas_train = np.mean(f_meas_train, axis=0)
        f_meas_validation = np.mean(f_meas_validation, axis=0)
        loss_train = np.mean(loss_train, axis=0)
        loss_validation = np.mean(loss_validation, axis=0)

        f_meas_train = f_meas_train[:, -1]
        f_meas_validation = f_meas_validation[:, -1]
        loss_train = loss_train[:, -1]
        loss_validation = loss_validation[:, -1]

        return f_meas_train, f_meas_validation, loss_train, loss_validation, \
               f_meas_train_std, f_meas_validation_std, loss_train_std, loss_validation_std

    def plot_all(self):

        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = .25
        markersize = 5.
        std_multiplier = 2
        label_fontsize = 18
        legend_fontsize = 14

        ista_marker= 'o'
        ista_color='blue'
        ista_label=r'\textsc{ista}'

        fista_marker= 'v'
        fista_color='red'
        fista_label=r'\textsc{fista}'

        freq_marker= 's'
        freq_color='black'
        freq_label=r'\textsc{lista}'

        sh_bayes_marker='d'
        sh_bayes_color='green'
        sh_bayes_label=r'Bayesian \textsc{lista}'

        plt.plot(self.K_array, self.ista_f_meas_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.ista_f_meas_train, yerr=std_multiplier * self.ista_f_meas_train_std, color=ista_color)
        plt.plot(self.K_array, self.fista_f_meas_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.fista_f_meas_train, yerr=std_multiplier * self.fista_f_meas_train_std, color=fista_color)
        plt.plot(self.K_array, self.freq_f_meas_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.freq_f_meas_train, yerr=2 * self.freq_f_meas_train_std, color=freq_color)
        plt.plot(self.K_array, self.sh_bayes_f_meas_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.sh_bayes_f_meas_train, yerr=2 * self.sh_bayes_f_meas_train_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('results/f_measure_train.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_f_meas_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.ista_f_meas_validation, yerr=3 * self.ista_f_meas_validation_std, color=ista_color)
        plt.plot(self.K_array, self.fista_f_meas_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.fista_f_meas_validation, yerr=3 * self.fista_f_meas_validation_std, color=fista_color)
        plt.plot(self.K_array, self.freq_f_meas_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.freq_f_meas_validation, yerr=3 * self.freq_f_meas_validation_std, color=freq_color)
        plt.plot(self.K_array, self.sh_bayes_f_meas_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.sh_bayes_f_meas_validation, yerr=3 * self.sh_bayes_f_meas_validation_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('results/f_measure_validation.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_loss_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.ista_loss_train, yerr=3 * self.ista_loss_train_std, color=ista_color)
        plt.plot(self.K_array, self.fista_loss_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.fista_loss_train, yerr=3 * self.fista_loss_train_std, color=fista_color)
        plt.plot(self.K_array, self.freq_loss_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.freq_loss_train, yerr=3 * self.freq_loss_train_std, color=freq_color)
        plt.plot(self.K_array, self.sh_bayes_loss_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.sh_bayes_loss_train, yerr=3 * self.sh_bayes_loss_train_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.savefig('results/nmse_train.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_loss_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.ista_loss_validation, yerr=3 * self.ista_loss_validation_std, color=ista_color)
        plt.plot(self.K_array, self.fista_loss_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.fista_loss_validation, yerr=3 * self.fista_loss_validation_std, color=fista_color)
        plt.plot(self.K_array, self.freq_loss_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.freq_loss_validation, yerr=3 * self.freq_loss_validation_std, color=freq_color)
        plt.plot(self.K_array, self.sh_bayes_loss_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.K_array, self.sh_bayes_loss_validation, yerr=3 * self.sh_bayes_loss_validation_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'K', fontsize=label_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.savefig('results/nmse_validation.eps', format='eps')
        plt.show()

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

if __name__=='__main__':

    plt.rc('text', usetex=True)
    # plt.rc('figure', figsize=figsize(0.9))


    collector = UndersamplingExperimentReultsCollector()
    collector.collect_all()
    collector.plot_all()