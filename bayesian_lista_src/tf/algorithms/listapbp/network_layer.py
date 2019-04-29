import tensorflow as tf
from ..shared.soft_thresholding import soft_threshold
import numpy as np

def n_cdf(x):
    return 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))

def n_pdf(x, mu, sigma):
    return 1 / tf.sqrt(2 * np.pi * sigma) * tf.exp(-0.5 * (x - mu) ** 2 / sigma)

def log_student_pdf(x, mu, beta, nu):
    return tf.lgamma(0.5 * (nu + 1)) - (tf.lgamma(0.5 * nu) + 0.5 * tf.log(np.pi * nu * beta)) - 0.5 * (nu + 1) * tf.log(
        (1 + 1 / nu * (x - mu) ** 2 / beta))

def student_pdf(x, mu, beta, nu):
    return tf.exp(log_student_pdf(x, mu, beta, nu))


class Network_layer:

    def __init__(self, thr_lambda):
        self.thr_lambda = thr_lambda

    def compute_B(self, y, W_M, W_V):
        B_m = tf.tensordot(W_M, y, 1)
        B_v = tf.tensordot(W_V, tf.pow(y, 2), 1)
        return B_m, B_v

    def compute_D(self, z_w_prev, z_m_prev, z_v_prev, S_M, S_V):
        var_z_prev = (1 - z_w_prev) * z_v_prev + (1 - z_w_prev) * z_w_prev * (z_m_prev ** 2)
        D_m = tf.tensordot(S_M, (1 - z_w_prev) * z_m_prev, 1)
        D_v = tf.tensordot(tf.pow(S_M, 2), var_z_prev, 1) + tf.tensordot(S_V, ((1 - z_w_prev) ** 2) * (z_m_prev ** 2), 1) + \
              tf.tensordot(S_V, var_z_prev, 1)
        return D_m, D_v

    def compute_C(self, B_m, B_v, D_m, D_v):
        C_m = B_m + D_m
        C_v = B_v + D_v
        return C_m, C_v

    def compute_new_z(self, C_m, C_v):
        z_new_m = 1 / tf.sqrt(2 * np.pi) * tf.sqrt(C_v) * tf.exp(-((self.thr_lambda - C_m) ** 2) / (2 * C_v)) - \
                  1 / tf.sqrt(2 * np.pi) * tf.sqrt(C_v) * tf.exp(-((self.thr_lambda + C_m) ** 2) / (2 * C_v)) + \
                  (C_m - self.thr_lambda) * (1 - n_cdf((self.thr_lambda - C_m) / (tf.sqrt(C_v)))) + \
                  (C_m + self.thr_lambda) * n_cdf((-self.thr_lambda - C_m) / (tf.sqrt(C_v)))

        z_new_v = 1 / tf.sqrt(2 * np.pi) * tf.sqrt(C_v) * (C_m - self.thr_lambda) * tf.exp(
            -((self.thr_lambda - C_m) ** 2) / (2 * C_v)) + \
                  (C_v + (C_m - self.thr_lambda) ** 2) * (
                          1 - n_cdf((self.thr_lambda - C_m) / (tf.sqrt(C_v)))) - \
                  1 / tf.sqrt(2 * np.pi) * tf.sqrt(C_v) * (self.thr_lambda + C_m) * tf.exp(
            -((self.thr_lambda + C_m) ** 2) / (2 * C_v)) + \
                  (C_v + (C_m + self.thr_lambda) ** 2) * n_cdf(
            (-self.thr_lambda - C_m) / (tf.sqrt(C_v))) - z_new_m ** 2

        z_new_w = n_cdf((self.thr_lambda - C_m) / (tf.sqrt(C_v))) - n_cdf(
            (-self.thr_lambda - C_m) / (tf.sqrt(C_v)))

        return z_new_w, z_new_m, z_new_v

    def output_probabilistic(self, z_w_prev, z_m_prev, z_v_prev, y, W_M, W_V, S_M, S_V):
        B_m, B_v = self.compute_B(y, W_M, W_V)
        D_m, D_v = self.compute_D(z_w_prev, z_m_prev, z_v_prev, S_M, S_V)
        C_m, C_v = self.compute_C(B_m, B_v, D_m, D_v)
        z_new_w, z_new_m, z_new_v = self.compute_new_z(C_m, C_v)
        return z_new_w, z_new_m, z_new_v

    def output_deterministic(self, output_previous, y, W, S):
        B = tf.tensordot(W, y, 1)
        D = tf.tensordot(S, output_previous, 1)
        C = B + D
        out = soft_threshold(C, self.thr_lambda)

        return out
