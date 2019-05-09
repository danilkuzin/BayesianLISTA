from tqdm import trange

from tf.algorithms.listapbp.handler import SingleBayesianListaHandler
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

    num_epochs = 10
    N = 100
    batch_size=1

    tf.enable_eager_execution()

    data_generator = DataGenerator(D=D, K=K, sparsity=sparsity, beta_scale=beta_scale, noise_scale=noise_scale)
    handler = SingleBayesianListaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), initial_lambda=initial_lambda)

    beta_train, y_train, _ = data_generator.new_sample(N)

    train_data = tf.data.Dataset.from_tensor_slices((beta_train, y_train)).shuffle(10).batch(batch_size=batch_size)


    beta, y, Noise = data_generator.new_sample(N)
    beta_valid, y_valid, _ = data_generator.new_sample(N)
    loss_hist, valid_loss_hist = [], []

    t = trange(num_epochs)
    for _ in t:
        for i, (beta_batch, y_batch) in enumerate(train_data):
            handler.train(num_epochs=num_epochs, beta_train=beta_batch, y_train=y_batch)
        loss = SingleBayesianListaHandler.loss(handler.predict(y.astype(np.float32)), beta)
        valid_loss = SingleBayesianListaHandler.loss(handler.predict(y_valid.astype(np.float32)), beta_valid)
        t.set_description(f'ML (loss={loss.numpy()}, valid_loss={valid_loss.numpy()})')
        loss_hist.append(loss.numpy())
        valid_loss_hist.append(valid_loss.numpy())

    plt.plot(loss_hist)
    plt.plot(valid_loss_hist)
    plt.show()

    beta_pred = handler.predict(y_valid.astype(np.float32))
    print(f'beta_valid_0: {beta_valid[0]}, beta_pred_0:{beta_pred[0]}')
    print(f'beta_valid_10: {beta_valid[10]}, beta_pred_10:{beta_pred[10]}')