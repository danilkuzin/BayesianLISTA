compare_nmist_script:

    def train_iteration(self):

        # self.freq_train_loss.append(
        #     self.freq_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))
        # self.bayesian_train_loss.append(
        #     self.bayesian_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))
        # self.shared_bayesian_train_loss.append(
        #     self.shared_bayesian_lista.train_iteration(beta_train=self.data.train_data, y_train=self.data.y_train))

        self.freq_train_loss.append(
            self.freq_lista.train_iteration_nmse(beta_train=self.data.train_data, y_train=self.data.y_train))
        #self.bayesian_train_loss.append(
        #    self.bayesian_lista.train_iteration_nmse(beta_train=self.data.train_data, y_train=self.data.y_train))
        self.shared_bayesian_train_loss.append(
self.shared_bayesian_lista.train_iteration_nmse(beta_train=self.data.train_data, y_train=self.data.y_train,
                                                            sample_mean=True))

    np.random.seed(1)

    K = 100
    L = 10

    # batch_size = 5000
    # validation_size = 100

    saved_comparator_file_name = []#'comparator_with_random_dictionary_4_iter.pkl'


    if not saved_comparator_file_name:
        comparator = MnistSequentialComparator(K, L, learning_rate=0.0001)
    else:
        comparator = pickle.load(open(saved_comparator_file_name, 'rb'))




    n_iter = 300

    for _ in tqdm(range(n_iter)):
        comparator.train_iteration()

mnist_data:

    self.training_size = 100
    self.validation_size = 100



learning_rate is constant for Lista, update for each data point

prior:
	__init__:
	n_samples = 3.0
        v_observed = 1.0
        #v_observed = 10000.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed
        self.a_s = 2.0 * n_samples
        self.b_s = 2.0 * n_samples * v_observed



