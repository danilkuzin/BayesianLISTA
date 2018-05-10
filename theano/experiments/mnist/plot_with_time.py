import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    data = np.load('time_100_train_20_layers_K_250.npz')

    plt.semilogy(data['lista_time'], data['lista_nmse'])
    plt.semilogy(data['sh_bayes_time'], data['sh_bayes_nmse'])
    plt.axis([0, 4000, 0.5, 1])
    plt.show()

    # data = np.load('time_100_train_20_layers_K_100.npz')
    #
    # plt.plot(data['lista_time'], data['lista_nmse'])
    # plt.plot(data['sh_bayes_time'], data['sh_bayes_nmse'])
    # plt.axis([0, 4000, 0.5, 1])
    # plt.show()