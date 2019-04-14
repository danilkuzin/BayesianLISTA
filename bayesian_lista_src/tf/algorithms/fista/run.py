from tqdm import trange

from tf.algorithms.fista.handler import FistaHandler
from tf.data.synthetic.data_generator import DataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    D = 10
    K = 5
    L = 5
    sparsity = 0.8
    beta_scale = 1
    noise_scale = 0.1
    learning_rate = 0.1
    initial_lambda = 0.2

    tf.enable_eager_execution()

    data_generator = DataGenerator(D=D, K=K, sparsity=sparsity, beta_scale=beta_scale, noise_scale=noise_scale)
    handler = IstaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), initial_lambda=initial_lambda)

    num_epochs = 10
    N = 1000
    beta, y, Noise = data_generator.new_sample(N)
    beta_valid, y_valid, _ = data_generator.new_sample(N)
    loss_hist, valid_loss_hist = [], []
    t = trange(100, desc='ML')
    for i in t:
        prediction = handler.predict(y.astype(np.float32))
        loss = FistaHandler.loss(handler.model(y.astype(np.float32)), beta)
        valid_loss = FistaHandler.loss(handler.model(y_valid.astype(np.float32)), beta_valid)
        t.set_description(f'ML (loss={loss.numpy()}, valid_loss={valid_loss.numpy()})')
        loss_hist.append(loss.numpy())
        valid_loss_hist.append(valid_loss.numpy())

    plt.plot(loss_hist)
    plt.plot(valid_loss_hist)
    plt.show()

    beta_pred = handler.model(y_valid.astype(np.float32))
    print(f'beta_valid_0: {beta_valid[0]}, beta_pred_0:{beta_pred[0]}')
    print(f'beta_valid_10: {beta_valid[10]}, beta_pred_10:{beta_pred[10]}')