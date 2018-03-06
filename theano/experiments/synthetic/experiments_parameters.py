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
    n_iter = 10
    batch_size = 500
    validation_size = 100
    return rseed, D, K, L, batch_size, validation_size, n_iter

def load_long_experiment():
    D = 100
    K = 50
    L = 4

    rseed = 1
    n_iter = 1000
    batch_size = 1000
    validation_size = 100
    return rseed, D, K, L, batch_size, validation_size, n_iter