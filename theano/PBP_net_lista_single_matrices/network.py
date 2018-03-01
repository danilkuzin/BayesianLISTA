import numpy as np

import theano

import theano.tensor as T

from PBP_net_lista_single_matrices import network_layer
from PBP_net_lista_single_matrices.network_layer import theano_soft_threshold


class Network:

    def __init__(self, W_M_init, W_V_init, S_M_init, S_V_init, thr_lambda_init, a_init, b_init, D, K, L):

        self.D = D  # size of beta
        self.K = K  # size of y

        self.params_W_M = theano.shared(value=W_M_init.astype(theano.config.floatX), name='W_M', borrow=True)
        self.params_W_V = theano.shared(value=W_V_init.astype(theano.config.floatX), name='W_V', borrow=True)
        self.params_W = theano.shared(value=W_V_init.astype(theano.config.floatX), name='W', borrow=True)
        self.params_S_M = theano.shared(value=S_M_init.astype(theano.config.floatX), name='S_M', borrow=True)
        self.params_S_V = theano.shared(value=S_V_init.astype(theano.config.floatX), name='S_V', borrow=True)
        self.params_S = theano.shared(value=S_V_init.astype(theano.config.floatX), name='S', borrow=True)

        self.layers = []

        for layer in range(L):
            self.layers.append(network_layer.Network_layer(thr_lambda_init))

        self.params_thr_lambda = []

        for layer in self.layers:
            self.params_thr_lambda.append(layer.thr_lambda)

        self.a = theano.shared(value=a_init)
        self.b = theano.shared(value=b_init)

    def output_deterministic(self, y):

        # Recursively compute output
        x = T.zeros((self.D,), dtype=np.float)

        for layer in self.layers:
            x = layer.output_deterministic(x, y, self.params_W, self.params_S)

        return x

    def output_probabilistic(self, y):

        v = T.zeros((self.D,), dtype=np.float)
        w = T.zeros((self.D,), dtype=np.float)
        m = T.zeros((self.D,), dtype=np.float)

        for layer in self.layers:
            w, m, v = layer.output_probabilistic(w, m, v, y, self.params_W_M, self.params_W_V, self.params_S_M, self.params_S_V)

        return w, m, v

    def logZ_Z1_Z2(self, beta, y):

        w, m, v = self.output_probabilistic(y)

        v_final = v + self.b / (self.a - 1) * T.ones(self.D)
        v_final1 = v + self.b / self.a * T.ones(self.D)
        v_final2 = v + self.b / (self.a + 1) * T.ones(self.D)

        mu_student = T.zeros(self.D)
        v_student = self.b / self.a * T.ones(self.D)
        v_student1 = self.b / (self.a + 1) * T.ones(self.D)
        v_student2 = self.b / (self.a + 2) * T.ones(self.D)

        nu_student = 2 * self.a * T.ones(self.D)
        nu_student1 = 2 * (self.a + 1) * T.ones(self.D)
        nu_student2 = 2 * (self.a + 2) * T.ones(self.D)

        logZ = T.sum(T.log(w * network_layer.student_pdf(beta, mu_student, v_student, nu_student)
            + (1 - w) * network_layer.n_pdf(beta, m, v_final)))
        logZ1 = T.sum(T.log(w * network_layer.student_pdf(beta, mu_student, v_student1, nu_student1)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final1)))
        logZ2 = T.sum(T.log(w * network_layer.student_pdf(beta, mu_student, v_student2, nu_student2)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final2)))

        return logZ, logZ1, logZ2

    def generate_updates(self, logZ, logZ1, logZ2):

        updates = []
        updates.append((self.params_W_M,
                        self.params_W_M + self.params_W_V * T.grad(logZ, self.params_W_M)))
        updates.append((self.params_W_V,
                        self.params_W_V - self.params_W_V ** 2 * (T.grad(logZ, self.params_W_M) ** 2 -
                                                                  2 * T.grad(logZ, self.params_W_V))))
        updates.append((self.params_S_M,
                        self.params_S_M + self.params_S_V * T.grad(logZ, self.params_S_M)))
        updates.append((self.params_S_V,
                        self.params_S_V - self.params_S_V ** 2 * (T.grad(logZ, self.params_S_M) ** 2 -
                                                                  2 * T.grad(logZ, self.params_S_V))))

        # updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * (self.a + 1) / self.a - 1.0)))
        # updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / self.b - T.exp(logZ1 - logZ) * self.a / self.b)))

        return updates

    def generate_updates_full_learning(self, logZ, logZ1, logZ2):

        updates = []
        updates.append((self.params_W_M,
                        self.params_W_M + self.params_W_V * T.grad(logZ, self.params_W_M)))
        updates.append((self.params_W_V,
                        self.params_W_V - self.params_W_V ** 2 * (T.grad(logZ, self.params_W_M) ** 2 -
                                                                  2 * T.grad(logZ, self.params_W_V))))
        updates.append((self.params_S_M,
                        self.params_S_M + self.params_S_V * T.grad(logZ, self.params_S_M)))
        updates.append((self.params_S_V,
                        self.params_S_V - self.params_S_V ** 2 * (T.grad(logZ, self.params_S_M) ** 2 -
                                                                  2 * T.grad(logZ, self.params_S_V))))

        updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / self.b - T.exp(logZ1 - logZ) * self.a / self.b)))

        return updates

    def get_params(self):

        return {'W_M': self.params_W_M.get_value(), 'W_V': self.params_W_V.get_value(),
                'S_M': self.params_S_M.get_value(), 'S_V': self.params_S_V.get_value(), 'a': self.a.get_value(),
                'b': self.b.get_value()}

    def set_params(self, params):

        self.params_W_M.set_value(params['W_M'])
        self.params_W_V.set_value(params['W_V'])
        self.params_S_M.set_value(params['S_M'])
        self.params_S_V.set_value(params['S_V'])

        self.a.set_value(params['a'])
        self.b.set_value(params['b'])

    def remove_invalid_updates(self, new_params, old_params):

        W_M_new = new_params['W_M']
        W_V_new = new_params['W_V']
        W_M_old = old_params['W_M']
        W_V_old = old_params['W_V']

        S_M_new = new_params['S_M']
        S_V_new = new_params['S_V']
        S_M_old = old_params['S_M']
        S_V_old = old_params['S_V']


        index1 = np.where(W_V_new <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(W_M_new),
                                        np.isnan(W_V_new)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            W_M_new[index] = W_M_old[index]
            W_V_new[index] = W_V_old[index]


        index1 = np.where(S_V_new <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(S_M_new),
                                        np.isnan(S_V_new)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            S_M_new[index] = S_M_old[index]
            S_V_new[index] = S_V_old[index]

    def sample_ws(self):

        W = self.params_W_M.get_value() \
            + np.random.randn(self.params_W_M.get_value().shape[0], self.params_W_M.get_value().shape[1]) \
            * np.sqrt(self.params_W_V.get_value())

        S = self.params_S_M.get_value() \
            + np.random.randn(self.params_S_M.get_value().shape[0],self.params_S_M.get_value().shape[1])\
            * np.sqrt(self.params_S_V.get_value())

        self.params_W.set_value(W)
        self.params_S.set_value(S)
