class ListaHandler(object):
    def __init__(self, D, K, L, X, initial_lambda):
        self.D = D
        self.K = K
        self.L = L
        self.X = X
        self.initial_lambda = initial_lambda

    def train_iteration(self, beta_train, y_train):
        raise NotImplementedError

    def test(self, beta_test, y_test):
        raise NotImplementedError

    def predict(self, y_test):
        raise NotImplementedError