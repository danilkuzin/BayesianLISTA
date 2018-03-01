class ListaHandler(object):
    def __init__(self, D, K, L, X):
        self.D = D
        self.K = K
        self.L = L
        self.X = X

    def train_iteration(self, beta_train, y_train):
        raise NotImplementedError

    def test(self, beta_test, y_test):
        raise NotImplementedError

    def predict(self, y_test):
        raise NotImplementedError