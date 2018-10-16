compare_nmist_script:

    np.random.seed(1)

    K = 100
    L = 2

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
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed
        self.a_s = 2.0 * n_samples
        self.b_s = 2.0 * n_samples * v_observed

        # We refine the factor for the prior variance on the noise

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets



