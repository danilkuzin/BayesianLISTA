def load_synthetic_experiment_1():
    rseed = 1
    D = 100
    K = 50
    L = 4
    batch_size = 500
    validation_size = 10
    n_iter = 100
    return rseed, D, K, L, batch_size, validation_size, n_iter

def load_synthetic_experiment_2():
    rseed = 1
    D = 100
    K = 20
    L = 4
    batch_size = 5000
    validation_size = 100
    n_iter = 10
    return rseed, D, K, L, batch_size, validation_size, n_iter

def load_quick_experiment():
    D = 10
    K = 8
    L = 4

    rseed = 1
    n_epochs = 50
    validation_size = 100

    sparsity = 0.8
    beta_scale = 1
    noise_scale = 0.1
    learning_rate = 0.1
    initial_lambda = 0.2
    batch_size = 25

    n_train = 500
    n_validation = 100

    return rseed, D, K, L, batch_size, validation_size, n_epochs, sparsity, beta_scale, noise_scale, n_train, n_validation, learning_rate, initial_lambda

def load_long_experiment():
    D = 100
    K = 50
    L = 4

    rseed = 1
    n_epochs = 1000
    n_train = 1000
    n_validation = 500
    batch_size = 5
    validation_size = 100
    return rseed, D, K, L, batch_size, validation_size, n_epochs, n_train, n_validation