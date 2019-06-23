import time

from tqdm import trange

from tf.algorithms.lista.handler import ListaHandler
from tf.data.synthetic.data_generator import DataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf.experiments.synthetic.experiments_parameters import load_quick_experiment

tf.enable_eager_execution()
rseed, D, K, L, batch_size, validation_size, n_epochs, sparsity, beta_scale, noise_scale, n_train, n_validation, learning_rate, initial_lambda = load_quick_experiment()
np.random.seed(rseed)
tf.set_random_seed(rseed)

data_generator = DataGenerator(D=D, K=K, sparsity=sparsity, beta_scale=beta_scale, noise_scale=noise_scale)
handler = ListaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), learning_rate=learning_rate,
                       initial_lambda=initial_lambda)

beta_train, y_train, Noise = data_generator.new_sample(n_train)
beta_valid, y_valid, _ = data_generator.new_sample(n_validation)

train_data = tf.data.Dataset.from_tensor_slices((beta_train, y_train)).shuffle(10).batch(batch_size=batch_size)

loss_hist, valid_loss_hist, f_meas_hist, valid_f_meas_hist, times = [], [], [], [], []

handler.train(n_epochs, train_data)

plt.plot(loss_hist)
plt.plot(valid_loss_hist)
plt.show()

plt.plot(f_meas_hist)
plt.plot(valid_f_meas_hist)
plt.show()

plt.plot(np.cumsum(times))
plt.show()

beta_pred = handler.model(y_valid.astype(np.float32))
print(f'beta_valid_0: {beta_valid[0]}, beta_pred_0:{beta_pred[0]}')
print(f'beta_valid_10: {beta_valid[10]}, beta_pred_10:{beta_pred[10]}')
