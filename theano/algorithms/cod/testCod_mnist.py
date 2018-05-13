from sklearn.decomposition import SparseCoder

from algorithms.PBP_net_lista.DataGenerator import DataGenerator
from algorithms.cod.cod import Cod
import numpy as np

import matplotlib.pyplot as plt

from experiments.mnist_sparse_coding.mnist_data_sparse_coding import MnistDataSparseCoding

if __name__ == "__main__":
    np.random.seed(1)
    K = 784
    D = 1000
    data = MnistDataSparseCoding(D, train_size=1000, valid_size=1000)
    data.check_download(normalise=False)
    data.learn_dictionary()



    X = data.X.T

    data_generator = DataGenerator(D, K, sparsity=0.99, beta_scale=1, noise_scale=0.5)
    beta, _, _ = data_generator.new_sample(10)

    beta = np.squeeze(beta)
    y = np.dot(beta, X)

    # cod = Cod(L=np.nan, D=D, K=K, X=X, initial_lambda=0.05, threshold=1e-6)
    # beta_estimator = cod.predict_full(y=np.squeeze(y))

    coder = SparseCoder(dictionary=X, transform_algorithm='omp')
    beta_estimated_coder = np.squeeze(coder.transform(y))

    # y_estimator = np.dot(X, beta_estimator)
    y_estimated_coder = np.dot(beta_estimated_coder, X)

    plt.imshow(np.reshape(y[0], (28, 28)))
    plt.show()

    # plt.imshow(np.reshape(y_estimator[0], (28, 28)))
    # plt.show()

    plt.imshow(np.reshape(y_estimated_coder[0], (28, 28)))
    plt.show()



