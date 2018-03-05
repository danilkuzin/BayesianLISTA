
import numpy as np

class Prior:

    def __init__(self, L, D, K, var_targets):

        # We refine the factor for the prior variance on the weights

        n_samples = 3.0
        #v_observed = 1.0
        v_observed = 10000.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed
        self.a_s = 2.0 * n_samples
        self.b_s = 2.0 * n_samples * v_observed

        # We refine the factor for the prior variance on the noise

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        # We refine the gaussian prior on the weights

        self.rnd_m_w = np.random.randn(D, K)
        self.m_w_hat_nat = np.zeros((D, K))
        self.v_w_hat_nat = (self.a_w - 1) / self.b_w * np.ones((D, K))
        self.a_w_hat_nat = np.zeros((D, K))
        self.b_w_hat_nat = np.zeros((D, K))

        self.rnd_m_s = np.random.randn(D, D)
        self.m_s_hat_nat = np.zeros((D, D))
        self.v_s_hat_nat = (self.a_s - 1) / self.b_s * np.ones((D, D))
        self.a_s_hat_nat = np.zeros((D, D))
        self.b_s_hat_nat = np.zeros((D, D))

    def get_initial_params(self):

        m_w = self.rnd_m_w
        v_w = 1.0 / self.v_w_hat_nat
        m_s = self.rnd_m_s
        v_s = 1.0 / self.v_s_hat_nat

        return {'W_M': m_w, 'W_V': v_w, 'S_M': m_s, 'S_V': v_s, 'a': self.a_sigma_hat_nat + 1,
                'b': -self.b_sigma_hat_nat}

    def get_params(self):

        m_w = self.m_w_hat_nat / self.v_w_hat_nat
        v_w = 1.0 / self.v_w_hat_nat
        m_s = self.m_s_hat_nat / self.v_s_hat_nat
        v_s = 1.0 / self.v_s_hat_nat

        return {'W_M': m_w, 'W_V': v_w, 'S_M': m_s, 'S_V': v_s, 'a': self.a_sigma_hat_nat + 1,
                'b': -self.b_sigma_hat_nat}

    def refine_prior(self, params):

        for j in range(params['W_M'].shape[0]):
            for k in range(params['W_M'].shape[1]):

                # We obtain the parameters of the cavity distribution

                v_w_nat = 1.0 / params['W_V'][j, k]
                m_w_nat = params['W_M'][j, k] / \
                    params['W_V'][j, k]
                v_w_cav_nat = v_w_nat - self.v_w_hat_nat[j, k]   # (36)
                m_w_cav_nat = m_w_nat - self.m_w_hat_nat[j, k]   # (36)
                v_w_cav = 1.0 / v_w_cav_nat
                m_w_cav = m_w_cav_nat / v_w_cav_nat
                a_w_nat = self.a_w - 1
                b_w_nat = -self.b_w
                a_w_cav_nat = a_w_nat - self.a_w_hat_nat[j, k]
                b_w_cav_nat = b_w_nat - self.b_w_hat_nat[j, k]
                a_w_cav = a_w_cav_nat + 1
                b_w_cav = -b_w_cav_nat

                if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
                    v_w_cav < 1e6:

                    # We obtain the values of the new parameters of the
                    # posterior approximation

                    v = v_w_cav + b_w_cav / (a_w_cav - 1)
                    v1 = v_w_cav + b_w_cav / a_w_cav
                    v2 = v_w_cav + b_w_cav / (a_w_cav + 1)
                    logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
                    logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
                    logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
                    d_logZ_d_m_w_cav = -m_w_cav / v
                    d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav**2 / v**2
                    m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav
                    v_w_new = v_w_cav - v_w_cav**2 * \
                        (d_logZ_d_m_w_cav**2 - 2 * d_logZ_d_v_w_cav)
                    a_w_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
                        (a_w_cav + 1) / a_w_cav - 1.0)
                    b_w_new = 1.0 / (np.exp(logZ2 - logZ1) * \
                        (a_w_cav + 1) / (b_w_cav) - np.exp(logZ1 - \
                        logZ) * a_w_cav / b_w_cav)
                    v_w_new_nat = 1.0 / v_w_new
                    m_w_new_nat = m_w_new / v_w_new
                    a_w_new_nat = a_w_new - 1
                    b_w_new_nat = -b_w_new

                    # We update the parameters of the approximate factor,
                    # whih is given by the ratio of the new posterior
                    # approximation and the cavity distribution

                    self.m_w_hat_nat[j, k] = m_w_new_nat - \
                        m_w_cav_nat
                    self.v_w_hat_nat[j, k] = v_w_new_nat - \
                        v_w_cav_nat
                    self.a_w_hat_nat[j, k] = a_w_new_nat - \
                        a_w_cav_nat
                    self.b_w_hat_nat[j, k] = b_w_new_nat - \
                        b_w_cav_nat

                    # We update the posterior approximation

                    params['W_M'][j, k] = m_w_new
                    params['W_V'][j, k] = v_w_new

                    self.a_w = a_w_new
                    self.b_w = b_w_new

        for j in range(params['S_M'].shape[ 0 ]):
            for k in range(params['S_M'].shape[ 1 ]):

                # We obtain the parameters of the cavity distribution

                v_s_nat = 1.0 / params['S_V'][j, k]
                m_s_nat = params['S_M'][j, k] / \
                    params['S_V'][j, k]
                v_s_cav_nat = v_s_nat - self.v_s_hat_nat[j, k]   # (36)
                m_s_cav_nat = m_s_nat - self.m_s_hat_nat[j, k]   # (36)
                v_s_cav = 1.0 / v_s_cav_nat
                m_s_cav = m_s_cav_nat / v_s_cav_nat
                a_s_nat = self.a_s - 1
                b_s_nat = -self.b_s
                a_s_cav_nat = a_s_nat - self.a_s_hat_nat[j, k]
                b_s_cav_nat = b_s_nat - self.b_s_hat_nat[j, k]
                a_s_cav = a_s_cav_nat + 1
                b_s_cav = -b_s_cav_nat

                if v_s_cav > 0 and b_s_cav > 0 and a_s_cav > 1 and \
                    v_s_cav < 1e6:

                    # We obtain the values of the new parameters of the
                    # posterior approximation

                    v = v_s_cav + b_s_cav / (a_s_cav - 1)
                    v1  = v_s_cav + b_s_cav / a_s_cav
                    v2  = v_s_cav + b_s_cav / (a_s_cav + 1)
                    logZ = -0.5 * np.log(v) - 0.5 * m_s_cav**2 / v
                    logZ1 = -0.5 * np.log(v1) - 0.5 * m_s_cav**2 / v1
                    logZ2 = -0.5 * np.log(v2) - 0.5 * m_s_cav**2 / v2
                    d_logZ_d_m_s_cav = -m_s_cav / v
                    d_logZ_d_v_s_cav = -0.5 / v + 0.5 * m_s_cav**2 / v**2
                    m_s_new = m_s_cav + v_s_cav * d_logZ_d_m_s_cav
                    v_s_new = v_s_cav - v_s_cav**2 * \
                        (d_logZ_d_m_s_cav**2 - 2 * d_logZ_d_v_s_cav)
                    a_s_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
                        (a_s_cav + 1) / a_s_cav - 1.0)
                    b_s_new = 1.0 / (np.exp(logZ2 - logZ1) * \
                        (a_s_cav + 1) / (b_s_cav) - np.exp(logZ1 - \
                        logZ) * a_s_cav / b_s_cav)
                    v_s_new_nat = 1.0 / v_s_new
                    m_s_new_nat = m_s_new / v_s_new
                    a_s_new_nat = a_s_new - 1
                    b_s_new_nat = -b_s_new

                    # We update the parameters of the approximate factor,
                    # whih is given by the ratio of the new posterior
                    # approximation and the cavity distribution

                    self.m_s_hat_nat[j, k] = m_s_new_nat - \
                        m_s_cav_nat
                    self.v_s_hat_nat[j, k] = v_s_new_nat - \
                        v_s_cav_nat
                    self.a_s_hat_nat[j, k] = a_s_new_nat - \
                        a_s_cav_nat
                    self.b_s_hat_nat[j, k] = b_s_new_nat - \
                        b_s_cav_nat

                    # We update the posterior approximation

                    params['S_M'][j, k] = m_s_new
                    params['S_V'][j, k] = v_s_new

                    self.a_s = a_s_new
                    self.b_s = b_s_new

        return params
