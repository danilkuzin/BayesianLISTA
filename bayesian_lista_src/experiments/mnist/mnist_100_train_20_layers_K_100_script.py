import numpy as np
from tqdm import tqdm

from experiments.mnist.mnist_sequential_comparator import MnistSequentialComparator

if __name__ == '__main__':

    np.random.seed(1)

    K = 100
    L = 20

    comparator = MnistSequentialComparator(K, L, learning_rate=0.0001, n_train_sample=100, n_validation_sample=100,
                                           train_freq=True, train_bayes=False, train_shared_bayes=True, use_ista=True,
                                           use_fista=True, save_history=False, initial_lambda=0.1)

    n_iter = 500

    for _ in tqdm(range(n_iter)):
        comparator.train_iteration()

    lista_time = comparator.recorders['lista'].time
    sh_bayes_time = comparator.recorders['shared_bayes'].time
    lista_nmse = comparator.recorders['lista'].validation_loss
    lista_f_meas = comparator.recorders['lista'].validation_f_meas
    sh_bayes_nmse = comparator.recorders['shared_bayes'].validation_loss
    sh_bayes_f_meas = comparator.recorders['shared_bayes'].validation_f_meas
    ista_time = comparator.recorders['ista'].time
    fista_time = comparator.recorders['fista'].time
    ista_nmse = comparator.recorders['ista'].validation_loss
    fista_nmse = comparator.recorders['fista'].validation_loss
    ista_f_meas = comparator.recorders['ista'].validation_f_meas
    fista_f_meas = comparator.recorders['fista'].validation_f_meas

    np.savez('time_100_train_20_layers_K_100',
             lista_time=lista_time,
             sh_bayes_time=sh_bayes_time,
             lista_nmse=lista_nmse,
             lista_f_meas=lista_f_meas,
             sh_bayes_nmse=sh_bayes_nmse,
             sh_bayes_f_meas=sh_bayes_f_meas,
             ista_time=ista_time,
             fista_time=fista_time,
             ista_nmse=ista_nmse,
             fista_nmse=fista_nmse,
             ista_f_meas=ista_f_meas,
             fista_f_meas=fista_f_meas
             )
