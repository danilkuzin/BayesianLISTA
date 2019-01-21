from tf.algorithms.lista.handler import ListaHandler
from tf.data.synthetic.data_generator import DataGenerator
import tensorflow as tf
import numpy as np

if __name__=="__main__":
    D = 10
    K = 5
    L = 2
    sparsity = 0.8
    beta_scale = 1
    noise_scale = 0.1
    learning_rate = 0.1
    initial_lambda = 0.2

    tf.enable_eager_execution()

    data_generator = DataGenerator(D=D, K=K, sparsity=sparsity, beta_scale=beta_scale, noise_scale=noise_scale)
    handler = ListaHandler(D=D, K=K, L=L, X=data_generator.X.astype(np.float32), learning_rate=learning_rate, initial_lambda=initial_lambda)

    num_epochs = 10
    N = 100
    beta, y, Noise = data_generator.new_sample(N)
    handler.train(num_epochs=10, beta_train=beta.astype(np.float32), y_train=y.astype(np.float32))
    #prediction = handler.predict(y.astype(np.float32))
    #loss = ListaHandler.loss(handler.model(y.astype(np.float32)), beta)
    #print(loss.numpy())