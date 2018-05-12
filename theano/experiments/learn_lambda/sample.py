from nonlinearities.soft_threshold import soft_threshold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_sample(lambda_shape, lambda_scale, gaussian_mean, gaussian_var):
    """
    Get sample of soft thresholding of Gaussian with parameter lambda that has Gamma prior
    :param lambda_shape:
    :param lambda_scale:
    :param gaussian_mean:
    :param gaussian_var:
    :return:
    """
    lambda_thr = np.random.gamma(lambda_shape, lambda_scale)
    x = np.random.normal(gaussian_mean, np.sqrt(gaussian_var))
    return soft_threshold(x, lambda_thr)

def get_samples_distribution(sample_size, lambda_shape, lambda_scale, gaussian_mean, gaussian_var):
    samples = np.zeros(sample_size)
    for i in range(sample_size):
        samples[i] = get_sample(lambda_shape, lambda_scale, gaussian_mean, gaussian_var)

    return samples

def exp_1():
    samples = get_samples_distribution(sample_size=10000, lambda_shape=1, lambda_scale=1, gaussian_mean=1, gaussian_var=1)
    sns.distplot(samples)
    plt.show()

def exp_2():
    samples = get_samples_distribution(sample_size=10000, lambda_shape=2, lambda_scale=2, gaussian_mean=1, gaussian_var=1)
    sns.distplot(samples)
    plt.show()

def exp_3():
    samples = get_samples_distribution(sample_size=10000, lambda_shape=1, lambda_scale=1, gaussian_mean=2, gaussian_var=2)
    sns.distplot(samples)
    plt.show()

def exp_4():
    samples = get_samples_distribution(sample_size=10000, lambda_shape=1, lambda_scale=1, gaussian_mean=0, gaussian_var=.1)
    sns.distplot(samples)
    plt.show()

def exp_5():
    samples = get_samples_distribution(sample_size=10000, lambda_shape=5, lambda_scale=5, gaussian_mean=5, gaussian_var=5)
    sns.distplot(samples)
    plt.show()

if __name__=="__main__":
    exp_1()