compare_nmist_script:

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




    n_iter = 10

    for _ in tqdm(range(n_iter)):
        comparator.train_iteration()

mnist_data:

    self.training_size = 500
    self.validation_size = 100