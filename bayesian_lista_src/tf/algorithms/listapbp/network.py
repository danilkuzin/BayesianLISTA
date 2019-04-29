import numpy as np

import tensorflow as tf

from ..listapbp import network_layer
from ..shared.soft_thresholding import soft_threshold


class Network:

    def __init__(self, W_M_init, W_V_init, S_M_init, S_V_init, thr_lambda_init, a_init, b_init, D, K, L):

        self.D = D  # size of beta
        self.K = K  # size of y

        self.params_W_M = tf.Variable(initial_value=W_M_init, name='W_M', dtype=tf.float32)
        self.params_W_V = tf.Variable(initial_value=W_V_init, name='W_V', dtype=tf.float32)
        self.params_W = tf.Variable(initial_value=W_V_init, name='W', dtype=tf.float32)
        self.params_S_M = tf.Variable(initial_value=S_M_init, name='S_M', dtype=tf.float32)
        self.params_S_V = tf.Variable(initial_value=S_V_init, name='S_V', dtype=tf.float32)
        self.params_S = tf.Variable(initial_value=S_V_init, name='S', dtype=tf.float32)

        self.layers = []

        for layer in range(L):
            self.layers.append(network_layer.Network_layer(thr_lambda_init))

        self.params_thr_lambda = []

        for layer in self.layers:
            self.params_thr_lambda.append(layer.thr_lambda)

        self.a = tf.Variable(initial_value=a_init, dtype=tf.float32)
        self.b = tf.Variable(initial_value=b_init, dtype=tf.float32)

    def output_deterministic(self, y):

        # Recursively compute output
        x = tf.zeros((self.D,), dtype=np.float)

        for layer in self.layers:
            x = layer.output_deterministic(x, y, self.params_W, self.params_S)

        return x

    def output_probabilistic(self, y):

        v = tf.zeros((self.D,), dtype=tf.float32)
        w = tf.zeros((self.D,), dtype=tf.float32)
        m = tf.zeros((self.D,), dtype=tf.float32)

        for layer in self.layers:
            w, m, v = layer.output_probabilistic(w, m, v, y, self.params_W_M, self.params_W_V, self.params_S_M, self.params_S_V)

        return w, m, v

    def logZ_Z1_Z2(self, beta, y):

        w, m, v = self.output_probabilistic(y)

        v_final = v + self.b / (self.a - 1) * tf.ones(self.D)
        v_final1 = v + self.b / self.a * tf.ones(self.D)
        v_final2 = v + self.b / (self.a + 1) * tf.ones(self.D)

        mu_student = tf.zeros(self.D)
        v_student = self.b / self.a * tf.ones(self.D)
        v_student1 = self.b / (self.a + 1) * tf.ones(self.D)
        v_student2 = self.b / (self.a + 2) * tf.ones(self.D)

        nu_student = 2 * self.a * tf.ones(self.D)
        nu_student1 = 2 * (self.a + 1) * tf.ones(self.D)
        nu_student2 = 2 * (self.a + 2) * tf.ones(self.D)

        logZ = tf.reduce_sum(tf.log(w * network_layer.student_pdf(beta, mu_student, v_student, nu_student)
            + (1 - w) * network_layer.n_pdf(beta, m, v_final)))
        logZ1 = tf.reduce_sum(tf.log(w * network_layer.student_pdf(beta, mu_student, v_student1, nu_student1)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final1)))
        logZ2 = tf.reduce_sum(tf.log(w * network_layer.student_pdf(beta, mu_student, v_student2, nu_student2)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final2)))

        return logZ, logZ1, logZ2

    def generate_updates(self, logZ, logZ1, logZ2, t):

        updates = []

        grads = t.gradient(logZ, [self.params_W_M, self.params_W_V, self.params_S_M, self.params_S_V])
        grad_WM, grad_WV, grad_SM, grad_SV = grads[0], grads[1], grads[2], grads[3]
        updated_WM = self.params_W_M + self.params_W_V * grad_WM
        updated_WV = self.params_W_V - self.params_W_V ** 2 * (grad_WM ** 2 - 2 * grad_WV)
        updated_SM = self.params_S_M + self.params_S_V * grad_SM
        updated_SV = self.params_S_V - self.params_S_V ** 2 * (grad_SM ** 2 - 2 * grad_SV)

        # updates.append((self.a, 1.0 / (tf.exp(logZ2 - 2 * logZ1 + logZ) * (self.a + 1) / self.a - 1.0)))
        # updates.append((self.b, 1.0 / (tf.exp(logZ2 - logZ1) * (self.a + 1) / self.b - tf.exp(logZ1 - logZ) * self.a / self.b)))

        #return updates
        # return {'W_M': self.params_W_M + self.params_W_V * grad_WM,
        #         'W_V': self.params_W_V - self.params_W_V ** 2 * (grad_WM ** 2 - 2 * grad_WV),
        #         'S_M': self.params_S_M + self.params_S_V * grad_SM,
        #         'S_V': self.params_S_V - self.params_S_V ** 2 * (grad_SM ** 2 - 2 * grad_SV)}

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

    def generate_updates_full_learning(self, logZ, logZ1, logZ2):

        updates = []
        updates.append((self.params_W_M,
                        self.params_W_M + self.params_W_V * tf.grad(logZ, self.params_W_M)))
        updates.append((self.params_W_V,
                        self.params_W_V - self.params_W_V ** 2 * (tf.grad(logZ, self.params_W_M) ** 2 -
                                                                  2 * tf.grad(logZ, self.params_W_V))))
        updates.append((self.params_S_M,
                        self.params_S_M + self.params_S_V * tf.grad(logZ, self.params_S_M)))
        updates.append((self.params_S_V,
                        self.params_S_V - self.params_S_V ** 2 * (tf.grad(logZ, self.params_S_M) ** 2 -
                                                                  2 * tf.grad(logZ, self.params_S_V))))

        updates.append((self.a, 1.0 / (tf.exp(logZ2 - 2 * logZ1 + logZ) * (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (tf.exp(logZ2 - logZ1) * (self.a + 1) / self.b - tf.exp(logZ1 - logZ) * self.a / self.b)))

        return updates

    def get_params(self):

        return {'W_M': self.params_W_M.numpy(), 'W_V': self.params_W_V.numpy(),
                'S_M': self.params_S_M.numpy(), 'S_V': self.params_S_V.numpy(), 'a': self.a.numpy(),
                'b': self.b.numpy()}

    def set_params(self, params):

        self.params_W_M.assign(params['W_M'])
        self.params_W_V.assign(params['W_V'])
        self.params_S_M.assign(params['S_M'])
        self.params_S_V.assign(params['S_V'])

        # self.a.assign(params['a'])
        # self.b.assign(params['b'])

    def apply_updates(self, new_params, old_params):

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

        self.params_W.assign(W)
        self.params_S.assign(S)

    def sample_mean_ws(self):

        W = self.params_W_M.numpy()

        S = self.params_S_M.numpy()

        self.params_W.assign(W)
        self.params_S.assign(S)