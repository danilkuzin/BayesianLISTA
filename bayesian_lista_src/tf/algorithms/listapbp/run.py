import time

from tqdm import trange

from tf.algorithms.listapbp.handler import SingleBayesianListaHandler
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
handler = SingleBayesianListaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), initial_lambda=initial_lambda)

beta_train, y_train, Noise = data_generator.new_sample(n_train)
beta_valid, y_valid, _ = data_generator.new_sample(n_validation)

batch_size = 1
train_data = tf.data.Dataset.from_tensor_slices((beta_train, y_train)).shuffle(10).batch(batch_size=batch_size)

loss_hist, valid_loss_hist, f_meas_hist, valid_f_meas_hist, times = [], [], [], [], []
n_epochs = 10
t = trange(n_epochs, desc='ML')
for _ in t:
    start_time = time.process_time()
    for i, (beta_batch, y_batch) in enumerate(train_data):
        handler.train_iteration(beta_train=beta_batch, y_train=y_batch)
    elapsed_time = time.process_time() - start_time

    train_pred = handler.predict(y_train.astype(np.float32))
    valid_pred = handler.predict(y_valid.astype(np.float32))
    loss = SingleBayesianListaHandler.loss(train_pred, beta_train)
    f_meas = SingleBayesianListaHandler.f_measure(beta_train, train_pred)
    valid_loss = SingleBayesianListaHandler.loss(valid_pred, beta_valid)
    valid_f_meas = SingleBayesianListaHandler.f_measure(beta_valid, valid_pred)
    t.set_description(
        f'ML (loss={loss.numpy():.3f}, valid_loss={valid_loss.numpy():.3f} f-meas:{f_meas:.3f} valid_f-meas:{valid_f_meas:.3f})')
    loss_hist.append(loss.numpy())
    valid_loss_hist.append(valid_loss.numpy())
    f_meas_hist.append(f_meas)
    valid_f_meas_hist.append(valid_f_meas)
    times.append(elapsed_time)

plt.plot(loss_hist)
plt.plot(valid_loss_hist)
plt.show()

beta_pred = handler.predict(y_valid.astype(np.float32))
print(f'beta_valid_0: {beta_valid[0]}, beta_pred_0:{beta_pred[0]}')
print(f'beta_valid_10: {beta_valid[10]}, beta_pred_10:{beta_pred[10]}')
