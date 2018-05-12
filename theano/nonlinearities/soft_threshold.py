import numpy as np

def soft_threshold(v, thr_lambda):
    return np.sign(v) * np.maximum(abs(v) - thr_lambda, np.zeros_like(v))