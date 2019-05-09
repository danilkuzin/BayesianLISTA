from tqdm import trange

from tf.algorithms.lista.handler import ListaHandler
from tf.data.synthetic.data_generator import DataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf.experiments.synthetic.experiments_parameters import load_quick_experiment

if __name__=="__main__":
    D = 10
    K = 5
    L = 4
    sparsity = 0.8
    beta_scale = 1
    noise_scale = 0.1
    learning_rate = 0.1
    initial_lambda = 0.2
    batch_size = 25

    rseed, D, K, L, batch_size, validation_size, n_iter = load_quick_experiment()
    tf.enable_eager_execution()

    data_generator = DataGenerator(D=D, K=K, sparsity=sparsity, beta_scale=beta_scale, noise_scale=noise_scale)
    handler = ListaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), learning_rate=learning_rate, initial_lambda=initial_lambda)

    num_epochs = 10
    N = 10000
    beta_train, y_train, Noise = data_generator.new_sample(N)
    beta_valid, y_valid, _ = data_generator.new_sample(N)

    train_data = tf.data.Dataset.from_tensor_slices((beta_train, y_train)).shuffle(10).batch(batch_size=batch_size)

    loss_hist, valid_loss_hist, f_meas_hist, valid_f_meas_hist = [], [], [], []
    t = trange(100, desc='ML')
    for i in t:
        for i, (beta_batch, y_batch) in enumerate(train_data):
            handler.train(num_epochs=1, beta_train=beta_batch, y_train=y_batch)
        train_pred = handler.model(y_train.astype(np.float32))
        valid_pred = handler.model(y_valid.astype(np.float32))
        loss = ListaHandler.loss(train_pred, beta_train)
        f_meas = ListaHandler.f_measure(beta_train, train_pred)
        valid_loss = ListaHandler.loss(valid_pred, beta_valid)
        valid_f_meas = ListaHandler.f_measure(beta_valid, valid_pred)
        t.set_description(f'ML (loss={loss.numpy():.3f}, valid_loss={valid_loss.numpy():.3f} f-meas:{f_meas:.3f} valid_f-meas:{valid_f_meas:.3f})')
        loss_hist.append(loss.numpy())
        valid_loss_hist.append(valid_loss.numpy())
        f_meas_hist.append(f_meas)
        valid_f_meas_hist.append(valid_f_meas)

    plt.plot(loss_hist)
    plt.plot(valid_loss_hist)
    plt.show()

    plt.plot(f_meas_hist)
    plt.plot(valid_f_meas_hist)
    plt.show()

    beta_pred = handler.model(y_valid.astype(np.float32))
    print(f'beta_valid_0: {beta_valid[0]}, beta_pred_0:{beta_pred[0]}')
    print(f'beta_valid_10: {beta_valid[10]}, beta_pred_10:{beta_pred[10]}')