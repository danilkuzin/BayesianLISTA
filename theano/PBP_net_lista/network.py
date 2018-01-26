import numpy as np

import theano

import theano.tensor as T

import network_layer


class Network:

    def __init__(self, W_M_init, W_V_init, S_M_init, S_V_init, thr_lambda_init, a_init, b_init, D, K):

        self.D = D  # size of beta
        self.K = K  # size of y
        self.layers = []

        for W_M, W_V, S_M, S_V in zip(W_M_init, W_V_init, S_M_init, S_V_init):
            self.layers.append(network_layer.Network_layer(W_M, W_V, S_M, S_V, thr_lambda_init))

        self.params_W_M = []
        self.params_W_V = []
        self.params_S_M = []
        self.params_S_V = []
        self.params_thr_lambda = []

        for layer in self.layers:
            self.params_W_M.append(layer.W_M)
            self.params_W_V.append(layer.W_V)
            self.params_S_M.append(layer.S_M)
            self.params_S_V.append(layer.S_V)
            self.params_thr_lambda.append(layer.thr_lambda)

        # TODO must be vector of dim(beta)
        # TODO 2 but we update it as scalar in generate updates, as Z are scalar
        self.a = theano.shared(value=a_init)
        self.b = theano.shared(value=b_init)

    def output_probabilistic(self, y):

        v = T.zeros((self.D,), dtype=np.float)
        w = T.zeros((self.D,), dtype=np.float)
        m = T.zeros((self.D,), dtype=np.float)

        for layer in self.layers:
            w, m, v = layer.output_probabilistic(w, m, v, y)

        return w, m, v

    def Z_Z1_Z2(self, beta, y):

        w, m, v = self.output_probabilistic(y)

        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)

        mu_student = T.zeros((self.D))
        v_student = self.b / self.a
        v_student1 = self.b / (self.a + 1)
        v_student2 = self.b / (self.a + 2)

        nu_student = 2 * self.a
        nu_student1 = 2 * (self.a + 1)
        nu_student2 = 2 * (self.a + 2)

        Z = T.prod(w * network_layer.student_pdf(beta, mu_student, v_student, nu_student)
            + (1 - w) * network_layer.n_pdf(beta, m, v_final))
        Z1 = T.prod(w * network_layer.student_pdf(beta, mu_student, v_student1, nu_student1)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final1))
        Z2 = T.prod(w * network_layer.student_pdf(beta, mu_student, v_student2, nu_student2)
             + (1 - w) * network_layer.n_pdf(beta, m, v_final2))

        return Z, Z1, Z2

    def generate_updates(self, Z, Z1, Z2):

        updates = []
        for i in range(len(self.params_W_M)):
            updates.append((self.params_W_M[i], self.params_W_M[i] + self.params_W_V[i] * T.grad(T.log(Z), self.params_W_M[i])))
            updates.append((self.params_W_V[i], self.params_W_V[i] - self.params_W_V[i] ** 2 * (T.grad(T.log(Z), self.params_W_M[i]) ** 2 - 2 * T.grad(T.log(Z), self.params_W_V[i]))))
            updates.append((self.params_S_M[i], self.params_S_M[i] + self.params_S_V[i] * T.grad(T.log(Z), self.params_S_M[i])))
            updates.append((self.params_S_V[i], self.params_S_V[i] - self.params_S_V[i] ** 2 * (T.grad(T.log(Z), self.params_S_M[i]) ** 2 - 2 * T.grad(T.log(Z), self.params_S_V[i]))))

        updates.append((self.a, 1.0 / (T.exp(T.log(Z2) - 2 * T.log(Z1) + T.log(Z)) * (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(T.log(Z2) - T.log(Z1)) * (self.a + 1) / self.b - T.exp(T.log(Z1) - T.log(Z)) * self.a / self.b)))

        return updates

    def get_params(self):

        W_M = []
        W_V = []
        S_M = []
        S_V = []
        for layer in self.layers:
            W_M.append(layer.W_M.get_value())
            W_V.append(layer.W_V.get_value())
            S_M.append(layer.S_M.get_value())
            S_V.append(layer.S_V.get_value())

        return {'W_M': W_M, 'W_V': W_V, 'S_M': S_M, 'S_V': S_V, 'a': self.a.get_value(),
                'b': self.b.get_value()}

    def set_params(self, params):

        for i in range(len(self.layers)):
            self.layers[i].W_M.set_value(params['W_M'][i])
            self.layers[i].W_V.set_value(params['W_V'][i])
            self.layers[i].S_M.set_value(params['S_M'][i])
            self.layers[i].S_V.set_value(params['S_V'][i])

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

        for i in range(len(self.layers)):
            index1 = np.where(W_V_new[i] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(W_M_new[i]),
                                            np.isnan(W_V_new[i])))

            index = [np.concatenate((index1[0], index2[0])),
                     np.concatenate((index1[1], index2[1]))]

            if len(index[0]) > 0:
                W_M_new[i][index] = W_M_old[i][index]
                W_V_new[i][index] = W_V_old[i][index]

        for i in range(len(self.layers)):
            index1 = np.where(S_V_new[i] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(S_M_new[i]),
                                            np.isnan(S_V_new[i])))

            index = [np.concatenate((index1[0], index2[0])),
                     np.concatenate((index1[1], index2[1]))]

            if len(index[0]) > 0:
                S_M_new[i][index] = S_M_old[i][index]
                S_V_new[i][index] = S_V_old[i][index]
