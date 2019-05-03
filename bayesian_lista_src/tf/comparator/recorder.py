import numpy as np
import time

class Recorder(object):
    def __init__(self, handler, save_history, name):
        self.handler = handler
        self.save_history = save_history
        self.name = name

        self.train_loss = []
        self.validation_loss = []
        self.train_f_meas = []
        self.validation_f_meas = []
        self.time = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        start_time = time.process_time()
        cur_train_loss, cur_train_f_meas = \
            self.handler.train_iteration(beta_train=beta_train, y_train=y_train)
        elapsed_time = time.process_time() - start_time
        if not self.time:
            self.time.append(elapsed_time)
        else:
            self.time.append(elapsed_time + self.time[-1])
        self.train_loss.append(cur_train_loss)
        self.train_f_meas.append(cur_train_f_meas)

        cur_valid_loss, cur_valid_f_meas = \
            self.handler.test(beta_test=beta_validation, y_test=y_validation)
        self.validation_loss.append(cur_valid_loss)
        self.validation_f_meas.append(cur_valid_f_meas)

    def save_numpy(self, filename):
        np.savez('{}_{}'.format(filename, self.name), D=self.handler.D, K=self.handler.K, L=self.handler.L,
                 train_loss=self.train_loss, validation_loss=self.validation_loss,
                 train_f_meas=self.train_f_meas, validation_f_meas=self.validation_f_meas)


class FrequentistRecorder(Recorder):
    pass


class IstaRecorder(FrequentistRecorder):
    def __init__(self, handler, name):
        save_history = False
        super().__init__(handler, save_history, name)


class FistaRecorder(FrequentistRecorder):
    def __init__(self, handler, name):
        save_history = False
        super().__init__(handler, save_history, name)


class BayesianRecorder(Recorder):
    pass


class SharedBayesianRecorder(BayesianRecorder):
    def __init__(self, handler, save_history, name):
        super().__init__(handler, save_history, name)
        if self.save_history:
            self.w_hist = []
            self.w_var_hist = []
            self.s_hist = []
            self.s_var_hist = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        super().train_and_record(beta_train, y_train, beta_validation, y_validation)

        if self.save_history:
            self.w_hist.append(
                self.handler.pbp_instance.network.params_W_M.get_value())
            self.w_var_hist.append(
                self.handler.pbp_instance.network.params_W_V.get_value())
            self.s_hist.append(
                self.handler.pbp_instance.network.params_S_M.get_value())
            self.s_var_hist.append(
                self.handler.pbp_instance.network.params_S_V.get_value())

    def save_numpy(self, filename):
        bayes_W_M = self.handler.pbp_instance.network.params_W_M.get_value()
        bayes_W_V = self.handler.pbp_instance.network.params_W_V.get_value()
        bayes_S_M = self.handler.pbp_instance.network.params_S_M.get_value()
        bayes_S_V = self.handler.pbp_instance.network.params_S_V.get_value()

        np.savez('{}_{}'.format(filename, self.name), D=self.handler.D, K=self.handler.K, L=self.handler.L,
                 train_loss=self.train_loss, validation_loss=self.validation_loss,
                 train_f_meas=self.train_f_meas, validation_f_meas=self.validation_f_meas,
                 bayes_S_M=bayes_S_M, bayes_S_V=bayes_S_V, bayes_W_M=bayes_W_M, bayes_W_V=bayes_W_V)


class FrequentistListaRecorder(FrequentistRecorder):
    def __init__(self, handler, save_history, name):
        super().__init__(handler, save_history, name)
        if self.save_history:
            self.w_hist = []
            self.s_hist = []

    def train_and_record(self, beta_train, y_train, beta_validation, y_validation):
        super().train_and_record(beta_train, y_train, beta_validation, y_validation)

        if self.save_history:
            self.w_hist.append(self.handler.lista.W.get_value())
            self.s_hist.append(self.handler.lista.S.get_value())