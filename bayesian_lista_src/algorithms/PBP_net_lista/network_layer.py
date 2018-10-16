import theano

import theano.tensor as T

import numpy as np


def n_cdf(x):
    return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))


def n_pdf(x, mu, sigma):
    return 1 / T.sqrt(2 * np.pi * sigma) * T.exp(-0.5 * (x - mu) ** 2 / sigma)


def student_pdf(x, mu, beta, nu):
    return T.gamma(0.5 * (nu + 1)) / (T.gamma(0.5 * nu) * T.sqrt(np.pi * nu * beta)) * T.power(
        (1 + 1 / nu * (x - mu) ** 2 / beta), -0.5 * (nu + 1))


def student_pdf_new(x, mu, beta, nu):
    return T.exp(T.gammaln(0.5 * (nu + 1)) - T.gammaln(0.5 * nu) - 0.5 * T.log(np.pi) - 0.5 * T.log(nu)
                 - 0.5 * T.log(beta) - 0.5 * (nu + 1) * T.log(1 + 1 / nu * (x - mu) ** 2 / beta))


def theano_soft_threshold(v, thr_lambda):
    return T.sgn(v) * T.maximum(abs(v) - thr_lambda, T.zeros_like(v))


class Network_layer:

    def __init__(self, W_M_init, W_V_init, S_M_init, S_V_init, thr_lambda):
        self.W_M = theano.shared(value=W_M_init.astype(theano.config.floatX), name='W_M', borrow=True)
        self.W_V = theano.shared(value=W_V_init.astype(theano.config.floatX), name='W_V', borrow=True)
        self.W = theano.shared(value=W_V_init.astype(theano.config.floatX), name='W', borrow=True)
        self.S_M = theano.shared(value=S_M_init.astype(theano.config.floatX), name='S_M', borrow=True)
        self.S_V = theano.shared(value=S_V_init.astype(theano.config.floatX), name='S_V', borrow=True)
        self.S = theano.shared(value=S_V_init.astype(theano.config.floatX), name='S', borrow=True)

        self.thr_lambda = thr_lambda

    def compute_B(self, y):
        B_m = T.dot(self.W_M, y)
        B_v = T.dot(self.W_V, y ** 2)
        return B_m, B_v

    def compute_D(self, z_w_prev, z_m_prev, z_v_prev):
        var_z_prev = (1 - z_w_prev) * z_v_prev + (1 - z_w_prev) * z_w_prev * (z_m_prev ** 2)
        D_m = T.dot(self.S_M, (1 - z_w_prev) * z_m_prev)
        D_v = T.dot(self.S_M ** 2, var_z_prev) + T.dot(self.S_V, ((1 - z_w_prev) ** 2) * (z_m_prev ** 2)) + \
              T.dot(self.S_V, var_z_prev)
        return D_m, D_v

    def compute_C(self, B_m, B_v, D_m, D_v):
        C_m = B_m + D_m
        C_v = B_v + D_v
        return C_m, C_v

    def compute_new_z(self, C_m, C_v):
        z_new_m = 1 / T.sqrt(2 * np.pi) * T.sqrt(C_v) * T.exp(-((self.thr_lambda - C_m) ** 2) / (2 * C_v)) - \
                  1 / T.sqrt(2 * np.pi) * T.sqrt(C_v) * T.exp(-((self.thr_lambda + C_m) ** 2) / (2 * C_v)) + \
                  (C_m - self.thr_lambda) * (1 - n_cdf((self.thr_lambda - C_m) / (T.sqrt(C_v)))) + \
                  (C_m + self.thr_lambda) * n_cdf((-self.thr_lambda - C_m) / (T.sqrt(C_v)))

        z_new_v = 1 / T.sqrt(2 * np.pi) * T.sqrt(C_v) * (C_m - self.thr_lambda) * T.exp(
            -((self.thr_lambda - C_m) ** 2) / (2 * C_v)) + \
                  (C_v + (C_m - self.thr_lambda) ** 2) * (
                          1 - n_cdf((self.thr_lambda - C_m) / (T.sqrt(C_v)))) - \
                  1 / T.sqrt(2 * np.pi) * T.sqrt(C_v) * (self.thr_lambda + C_m) * T.exp(
            -((self.thr_lambda + C_m) ** 2) / (2 * C_v)) + \
                  (C_v + (C_m + self.thr_lambda) ** 2) * n_cdf(
            (-self.thr_lambda - C_m) / (T.sqrt(C_v))) - z_new_m ** 2

        z_new_w = n_cdf((self.thr_lambda - C_m) / (T.sqrt(C_v))) - n_cdf(
            (-self.thr_lambda - C_m) / (T.sqrt(C_v)))

        return z_new_w, z_new_m, z_new_v

    def output_probabilistic(self, z_w_prev, z_m_prev, z_v_prev, y):
        B_m, B_v = self.compute_B(y)
        D_m, D_v = self.compute_D(z_w_prev, z_m_prev, z_v_prev)
        C_m, C_v = self.compute_C(B_m, B_v, D_m, D_v)
        z_new_w, z_new_m, z_new_v = self.compute_new_z(C_m, C_v)
        return z_new_w, z_new_m, z_new_v

    def output_deterministic(self, output_previous, y):
        B = T.dot(self.W, y)
        D = T.dot(self.S, output_previous)
        C = B + D
        out = theano_soft_threshold(C, self.thr_lambda)

        return out
