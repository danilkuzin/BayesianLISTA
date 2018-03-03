compare_sequential:

    np.random.seed(1)

    D = 784#100
    K = 100#50
    L = 4

    batch_size = 1000
    validation_size = 100

    saved_comparator_file_name = []#'best_model_bayes_lista_single_matrices.pkl'

    if not saved_comparator_file_name:
        comparator = SequentialComparator(D, K, L, learning_rate=0.0001, n_train_sample=batch_size, n_validation_sample=validation_size)
    else:
        comparator = pickle.load(open(saved_comparator_file_name, 'rb'))




    n_iter = 50 (+20 afterwards)

    for _ in tqdm(range(n_iter)):
        comparator.train_iteration()

