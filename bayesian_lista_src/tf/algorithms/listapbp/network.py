import numpy as np

import tensorflow as tf

from ..listapbp import tensor_network_layer
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
            self.layers.append(tensor_network_layer.Network_layer(thr_lambda_init))

        self.params_thr_lambda = []

        for layer in self.layers:
            self.params_thr_lambda.append(layer.thr_lambda)

        self.a = tf.Variable(initial_value=a_init, dtype=tf.float32)
        self.b = tf.Variable(initial_value=b_init, dtype=tf.float32)

    def output_deterministic(self, y):

        # Recursively compute output
        x = tf.zeros((y.shape[0], self.D), dtype=tf.float32)

        for layer in self.layers:
            x = layer.output_deterministic(x, y, self.params_W, self.params_S)

        return x

    def output_probabilistic(self, y):

        v = tf.zeros((y.shape[0], self.D), dtype=tf.float32)
        w = tf.zeros((y.shape[0], self.D), dtype=tf.float32)
        m = tf.zeros((y.shape[0], self.D), dtype=tf.float32)

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

        logZ = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student, nu_student)
            + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final)))
        logZ1 = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student1, nu_student1)
             + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final1)))
        logZ2 = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student2, nu_student2)
             + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final2)))

        return logZ, logZ1, logZ2

    def logZ_Z1_Z2_minibatch(self, beta, y):

        w, m, v = self.output_probabilistic(y)

        v_final = v + self.b / (self.a - 1) * tf.ones((y.shape[0], self.D))
        v_final1 = v + self.b / self.a * tf.ones((y.shape[0], self.D))
        v_final2 = v + self.b / (self.a + 1) * tf.ones((y.shape[0], self.D))

        mu_student = tf.zeros(self.D)
        v_student = self.b / self.a * tf.ones((y.shape[0], self.D))
        v_student1 = self.b / (self.a + 1) * tf.ones((y.shape[0], self.D))
        v_student2 = self.b / (self.a + 2) * tf.ones((y.shape[0], self.D))

        nu_student = 2 * self.a * tf.ones((y.shape[0], self.D))
        nu_student1 = 2 * (self.a + 1) * tf.ones((y.shape[0], self.D))
        nu_student2 = 2 * (self.a + 2) * tf.ones((y.shape[0], self.D))

        logZ = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student, nu_student)
            + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final)))
        logZ1 = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student1, nu_student1)
             + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final1)))
        logZ2 = tf.reduce_sum(tf.log(w * tensor_network_layer.student_pdf(beta, mu_student, v_student2, nu_student2)
             + (1 - w) * tensor_network_layer.n_pdf(beta, m, v_final2)))

        return logZ, logZ1, logZ2

    def generate_updates(self, logZ, logZ1, logZ2, t):

        grads = t.gradient(logZ, [self.params_W_M, self.params_W_V, self.params_S_M, self.params_S_V])
        grad_WM, grad_WV, grad_SM, grad_SV = grads[0], grads[1], grads[2], grads[3]
        updated_WM = (self.params_W_M + self.params_W_V * grad_WM).numpy()
        updated_WV = (self.params_W_V - self.params_W_V ** 2 * (grad_WM ** 2 - 2 * grad_WV)).numpy()
        updated_SM = (self.params_S_M + self.params_S_V * grad_SM).numpy()
        updated_SV = (self.params_S_V - self.params_S_V ** 2 * (grad_SM ** 2 - 2 * grad_SV)).numpy()

        # index = tf.where(tf.logical_or(tf.logical_or(
        #     tf.math.is_nan(updated_WM),
        #     tf.math.is_nan(updated_WV)),
        #     updated_WV <= 1e-100))
        #
        # if len(index) > 0:
        #     print(f"index:{index}")
        #     updated_WM[index] = self.params_W_M.numpy()[index]
        #     updated_WV[index] = self.params_W_V.numpy()[index]
        #
        # index = tf.where(tf.logical_or(tf.logical_or(
        #     tf.math.is_nan(updated_SM),
        #     tf.math.is_nan(updated_SV)),
        #     updated_SV <= 1e-100))
        #
        # if len(index) > 0:
        #     print(f"index:{index}")
        #     updated_SM[index] = self.params_S_M.numpy()[index]
        #     updated_SV[index] = self.params_S_V.numpy()[index]

        index1 = np.where(updated_WV <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(updated_WM),
                                        np.isnan(updated_WV)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            updated_WM[index] = self.params_W_M.numpy()[index]
            updated_WV[index] = self.params_W_V.numpy()[index]

        index1 = np.where(updated_SV <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(updated_SM),
                                        np.isnan(updated_SV)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            updated_SM[index] = self.params_S_M.numpy()[index]
            updated_SV[index] = self.params_S_V.numpy()[index]

        self.params_W_M.assign(updated_WM)
        self.params_W_V.assign(updated_WV)
        self.params_S_M.assign(updated_SM)
        self.params_S_V.assign(updated_SV)

    def generate_updates_minibatch(self, logZ, logZ1, logZ2, t):

        grads = t.gradient(logZ, [self.params_W_M, self.params_W_V, self.params_S_M, self.params_S_V])
        grad_WM, grad_WV, grad_SM, grad_SV = grads[0], grads[1], grads[2], grads[3]
        updated_WM = (self.params_W_M + self.params_W_V * grad_WM).numpy()
        updated_WV = (self.params_W_V - self.params_W_V ** 2 * (grad_WM ** 2 - 2 * grad_WV)).numpy()
        updated_SM = (self.params_S_M + self.params_S_V * grad_SM).numpy()
        updated_SV = (self.params_S_V - self.params_S_V ** 2 * (grad_SM ** 2 - 2 * grad_SV)).numpy()

        # index = tf.where(tf.logical_or(tf.logical_or(
        #     tf.math.is_nan(updated_WM),
        #     tf.math.is_nan(updated_WV)),
        #     updated_WV <= 1e-100))
        #
        # if len(index) > 0:
        #     print(f"index:{index}")
        #     updated_WM[index] = self.params_W_M.numpy()[index]
        #     updated_WV[index] = self.params_W_V.numpy()[index]
        #
        # index = tf.where(tf.logical_or(tf.logical_or(
        #     tf.math.is_nan(updated_SM),
        #     tf.math.is_nan(updated_SV)),
        #     updated_SV <= 1e-100))
        #
        # if len(index) > 0:
        #     print(f"index:{index}")
        #     updated_SM[index] = self.params_S_M.numpy()[index]
        #     updated_SV[index] = self.params_S_V.numpy()[index]

        index1 = np.where(updated_WV <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(updated_WM),
                                        np.isnan(updated_WV)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            updated_WM[index] = self.params_W_M.numpy()[index]
            updated_WV[index] = self.params_W_V.numpy()[index]

        index1 = np.where(updated_SV <= 1e-100)
        index2 = np.where(np.logical_or(np.isnan(updated_SM),
                                        np.isnan(updated_SV)))

        index = [np.concatenate((index1[0], index2[0])),
                 np.concatenate((index1[1], index2[1]))]

        if len(index[0]) > 0:
            updated_SM[index] = self.params_S_M.numpy()[index]
            updated_SV[index] = self.params_S_V.numpy()[index]

        self.params_W_M.assign(updated_WM)
        self.params_W_V.assign(updated_WV)
        self.params_S_M.assign(updated_SM)
        self.params_S_V.assign(updated_SV)

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