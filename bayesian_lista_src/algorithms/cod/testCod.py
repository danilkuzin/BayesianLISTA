from sklearn.decomposition import SparseCoder

from algorithms.PBP_net_lista.DataGenerator import DataGenerator
from algorithms.cod.cod import Cod
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)
    K = 8
    D = 10
    data_generator = DataGenerator(D, K, sparsity=0.8, beta_scale=1, noise_scale=0.0)
    beta, y, _ = data_generator.new_sample(1)
    beta = np.squeeze(beta)
    for k in range(data_generator.X.shape[1]):
        data_generator.X[:, k] = data_generator.X[:, k] / np.linalg.norm(data_generator.X[:, k], 2) ** 2
    y = np.dot(data_generator.X, beta)
    cod = Cod(L=np.nan, D=D, K=K, X=data_generator.X, initial_lambda=0.05, threshold=1e-6)
    beta_estimator = cod.predict_full(y=np.squeeze(y))

    coder = SparseCoder(dictionary=data_generator.X.T, transform_algorithm='threshold',
                        transform_alpha=0.05)
    beta_estimated_coder = coder.transform(y.reshape(1, -1))

    print("beta:{}, \n beta_estimator:{}, \n sparse coder estimate:{}".format(beta,
                                                                              beta_estimator, beta_estimated_coder))
