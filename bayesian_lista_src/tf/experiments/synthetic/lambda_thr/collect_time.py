import errno
import numpy as np
import os
import matplotlib.pyplot as plt
from experiments.synthetic.lambda_thr.collect_results import LambdaExperimentResultsCollector


class LambdaExperimentTimeCollector(LambdaExperimentResultsCollector):

    def collect_all(self):
        self.ista_f_meas_train, self.ista_f_meas_validation, self.ista_loss_train, self.ista_loss_validation, \
        self.ista_f_meas_train_std, self.ista_f_meas_validation_std, self.ista_loss_train_std, self.ista_loss_validation_std = \
            self.collect_main('ista')
        self.fista_f_meas_train, self.fista_f_meas_validation, self.fista_loss_train, self.fista_loss_validation, \
        self.fista_f_meas_train_std, self.fista_f_meas_validation_std, self.fista_loss_train_std, self.fista_loss_validation_std = \
            self.collect_main('fista')
        self.freq_f_meas_train, self.freq_f_meas_validation, self.freq_loss_train, self.freq_loss_validation, \
        self.freq_f_meas_train_std, self.freq_f_meas_validation_std, self.freq_loss_train_std, self.freq_loss_validation_std = \
            self.collect_main('freq')
        self.sh_bayes_f_meas_train, self.sh_bayes_f_meas_validation, self.sh_bayes_loss_train, self.sh_bayes_loss_validation,\
        self.sh_bayes_f_meas_train_std, self.sh_bayes_f_meas_validation_std, self.sh_bayes_loss_train_std, self.sh_bayes_loss_validation_std = \
            self.collect_main('sh_bayes')

        self.freq_loss_hist, self.freq_f_hist, self.freq_time_hist = self.collect_time_performance('freq', lambda_thr_ind=2)
        self.sh_bayes_loss_hist, self.sh_bayes_f_hist, self.sh_bayes_time_hist = self.collect_time_performance('sh_bayes', lambda_thr_ind=2)

    def collect_main(self, alg_name):
        f_meas_train = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))
        f_meas_validation = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))
        loss_train = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))
        loss_validation = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))

        for i in range(self.random_range):
            saved_res = np.load('{}/lambda_measures.npz'.format(i))
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

    def collect_time_performance(self, alg_name, lambda_thr_ind):
        f_meas = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))
        loss = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))
        time = np.zeros((self.random_range, len(self.lambda_array), self.n_iter))

        for i in range(self.random_range):
            saved_res = np.load('{}/lambda_measures.npz'.format(i))
            loss[i] = saved_res['{}_validation_loss'.format(alg_name)]
            f_meas[i] = saved_res['{}_validation_f_measure'.format(alg_name)]
            time[i] = saved_res['{}_time'.format(alg_name)]

        f_meas = f_meas[:, lambda_thr_ind, :]
        loss = loss[:, lambda_thr_ind, :]
        time = time[:, lambda_thr_ind, :]

        f_meas = np.mean(f_meas, axis=0)
        loss = np.mean(loss, axis=0)
        time = np.mean(time, axis=0)

        return loss, f_meas, time


    def plot_all(self):

        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        linewidth = .25
        markersize = 5.
        std_multiplier = 2
        label_fontsize = 24
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

        plt.semilogx(self.lambda_array, self.ista_f_meas_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.ista_f_meas_train, yerr=std_multiplier * self.ista_f_meas_train_std, color=ista_color)
        plt.semilogx(self.lambda_array, self.fista_f_meas_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.fista_f_meas_train, yerr=std_multiplier * self.fista_f_meas_train_std, color=fista_color)
        plt.semilogx(self.lambda_array, self.freq_f_meas_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.freq_f_meas_train, yerr=2 * self.freq_f_meas_train_std, color=freq_color)
        plt.semilogx(self.lambda_array, self.sh_bayes_f_meas_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.sh_bayes_f_meas_train, yerr=2 * self.sh_bayes_f_meas_train_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('results/f_measure_train.eps', format='eps')
        plt.show()

        plt.semilogx(self.lambda_array, self.ista_f_meas_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.ista_f_meas_validation, yerr=3 * self.ista_f_meas_validation_std, color=ista_color)
        plt.semilogx(self.lambda_array, self.fista_f_meas_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.fista_f_meas_validation, yerr=3 * self.fista_f_meas_validation_std, color=fista_color)
        plt.semilogx(self.lambda_array, self.freq_f_meas_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.freq_f_meas_validation, yerr=3 * self.freq_f_meas_validation_std, color=freq_color)
        plt.semilogx(self.lambda_array, self.sh_bayes_f_meas_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.sh_bayes_f_meas_validation, yerr=3 * self.sh_bayes_f_meas_validation_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('results/f_measure_validation.eps', format='eps')
        plt.show()

        plt.semilogx(self.lambda_array, self.ista_loss_train, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.ista_loss_train, yerr=3 * self.ista_loss_train_std, color=ista_color)
        plt.semilogx(self.lambda_array, self.fista_loss_train, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.fista_loss_train, yerr=3 * self.fista_loss_train_std, color=fista_color)
        plt.semilogx(self.lambda_array, self.freq_loss_train, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.freq_loss_train, yerr=3 * self.freq_loss_train_std, color=freq_color)
        plt.semilogx(self.lambda_array, self.sh_bayes_loss_train, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.sh_bayes_loss_train, yerr=3 * self.sh_bayes_loss_train_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.savefig('results/nmse_train.eps', format='eps')
        plt.show()

        plt.semilogx(self.lambda_array, self.ista_loss_validation, label=ista_label, color=ista_color, marker=ista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.ista_loss_validation, yerr=3 * self.ista_loss_validation_std, color=ista_color)
        plt.semilogx(self.lambda_array, self.fista_loss_validation, label=fista_label, color=fista_color, marker=fista_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.fista_loss_validation, yerr=3 * self.fista_loss_validation_std, color=fista_color)
        plt.semilogx(self.lambda_array, self.freq_loss_validation, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.freq_loss_validation, yerr=3 * self.freq_loss_validation_std, color=freq_color)
        plt.semilogx(self.lambda_array, self.sh_bayes_loss_validation, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.errorbar(self.lambda_array, self.sh_bayes_loss_validation, yerr=3 * self.sh_bayes_loss_validation_std, color=sh_bayes_color)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.savefig('results/nmse_validation.eps', format='eps')
        plt.show()

    def plot_time_performance(self):

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

        plt.plot(self.freq_time_hist, self.freq_f_hist, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.plot(self.sh_bayes_time_hist, self.sh_bayes_f_hist, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'F measure', fontsize=label_fontsize)
        plt.savefig('results/f_measure_time.eps', format='eps')
        plt.show()


        plt.semilogy(self.freq_time_hist, self.freq_loss_hist, label=freq_label, color=freq_color, marker=freq_marker, linewidth=linewidth, markersize=markersize)
        plt.semilogy(self.sh_bayes_time_hist, self.sh_bayes_loss_hist, label=sh_bayes_label, color=sh_bayes_color, marker=sh_bayes_marker, linewidth=linewidth, markersize=markersize)
        plt.legend(fontsize=legend_fontsize)
        plt.xlabel(r'L', fontsize=label_fontsize)
        plt.ylabel(r'\textsc{nmse}', fontsize=label_fontsize)
        plt.savefig('results/nmse_time.eps', format='eps')
        plt.show()


if __name__=='__main__':
    plt.rc('text', usetex=True)
    collector = LambdaExperimentTimeCollector()
    collector.collect_all()
    collector.plot_all()
    collector.plot_time_performance()
