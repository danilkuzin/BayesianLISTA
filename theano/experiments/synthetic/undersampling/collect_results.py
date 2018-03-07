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
        self.ista_f_meas_train, self.ista_f_meas_validation, self.ista_loss_train, self.ista_loss_validation = \
            self.collect_ista_fista('ista')
        self.fista_f_meas_train, self.fista_f_meas_validation, self.fista_loss_train, self.fista_loss_validation = \
            self.collect_ista_fista('fista')
        self.freq_f_meas_train, self.freq_f_meas_validation, self.freq_loss_train, self.freq_loss_validation = \
            self.collect_main('freq')
        self.sh_bayes_f_meas_train, self.sh_bayes_f_meas_validation, self.sh_bayes_loss_train, self.sh_bayes_loss_validation = \
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

        f_meas_train = np.mean(f_meas_train, axis=0)
        f_meas_validation = np.mean(f_meas_validation, axis=0)
        loss_train = np.mean(loss_train, axis=0)
        loss_validation = np.mean(loss_validation, axis=0)

        f_meas_train = f_meas_train[:, -1]
        f_meas_validation = f_meas_validation[:, -1]
        loss_train = loss_train[:, -1]
        loss_validation = loss_validation[:, -1]

        return f_meas_train, f_meas_validation, loss_train, loss_validation

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

        f_meas_train = np.mean(f_meas_train, axis=0)
        f_meas_validation = np.mean(f_meas_validation, axis=0)
        loss_train = np.mean(loss_train, axis=0)
        loss_validation = np.mean(loss_validation, axis=0)

        f_meas_train = f_meas_train[:, -1]
        f_meas_validation = f_meas_validation[:, -1]
        loss_train = loss_train[:, -1]
        loss_validation = loss_validation[:, -1]

        return f_meas_train, f_meas_validation, loss_train, loss_validation

    def plot_all(self):

        try:
            os.makedirs('results')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        plt.plot(self.K_array, self.ista_f_meas_train, label='ista')
        plt.plot(self.K_array, self.fista_f_meas_train, label='fista')
        plt.plot(self.K_array, self.freq_f_meas_train, label='freq lista')
        plt.plot(self.K_array, self.sh_bayes_f_meas_train, label='sh bayes')
        plt.legend()
        plt.title('F measure train')
        plt.savefig('results/f_measure_train.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_f_meas_validation, label='ista')
        plt.plot(self.K_array, self.fista_f_meas_validation, label='fista')
        plt.plot(self.K_array, self.freq_f_meas_validation, label='freq lista')
        plt.plot(self.K_array, self.sh_bayes_f_meas_validation, label='sh bayes')
        plt.legend()
        plt.title('F measure validation')
        plt.savefig('results/f_measure_validation.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_loss_train, label='ista')
        plt.plot(self.K_array, self.fista_loss_train, label='fista')
        plt.plot(self.K_array, self.freq_loss_train, label='freq lista')
        plt.plot(self.K_array, self.sh_bayes_loss_train, label='sh bayes')
        plt.legend()
        plt.title('NMSE train')
        plt.savefig('results/nmse_train.eps', format='eps')
        plt.show()

        plt.plot(self.K_array, self.ista_loss_validation, label='ista')
        plt.plot(self.K_array, self.fista_loss_validation, label='fista')
        plt.plot(self.K_array, self.freq_loss_validation, label='freq lista')
        plt.plot(self.K_array, self.sh_bayes_loss_validation, label='sh bayes')
        plt.legend()
        plt.title('NMSE validation')
        plt.savefig('results/nmse_validation.eps', format='eps')
        plt.show()


if __name__=='__main__':
    collector = UndersamplingExperimentReultsCollector()
    collector.collect_all()
    collector.plot_all()