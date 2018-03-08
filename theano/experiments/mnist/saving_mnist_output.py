freq_train_loss = comparator.freq_train_loss
freq_validation_loss = comparator.freq_validation_loss
shared_bayesian_train_loss = comparator.shared_bayesian_train_loss
shared_bayesian_validation_loss = comparator.shared_bayesian_validation_loss
freq_train_f_measure = comparator.freq_train_f_meas
freq_validation_f_measure = comparator.freq_validation_f_meas
shared_bayesian_f_measure = comparator.shared_bayesian_train_f_meas
shared_bayesian_train_f_measure = comparator.shared_bayesian_train_f_meas
shared_bayesian_validation_f_measure = comparator.shared_bayesian_validation_f_meas
freq_beta_train_est = comparator.freq_lista.predict(comparator.data.y_train)
freq_beta_validation_est = comparator.freq_lista.predict(comparator.data.y_validation)
shared_beta_train_est = comparator.shared_bayesian_lista.predict(comparator.data.y_train)
shared_beta_validation_est = comparator.shared_bayesian_lista.predict(comparator.data.y_validation)
true_beta_train = comparator.data.train_data
true_beta_validation = comparator.data.validation_data
y_train = comparator.data.y_train
y_validation = comparator.data.y_validation
D = comparator.data.train_data.shape[1]
train_size = comparator.data.train_data.shape[0]
bayes_W_M = comparator.shared_bayesian_lista.pbp_instance.network.params_W_M.get_value()
bayes_W_V = comparator.shared_bayesian_lista.pbp_instance.network.params_W_V.get_value()
bayes_S_M = comparator.shared_bayesian_lista.pbp_instance.network.params_S_M.get_value()
bayes_S_V = comparator.shared_bayesian_lista.pbp_instance.network.params_S_V.get_value()
np.savez('mnist_100_train_20_layers_K_250_bayes_weights', D = D, K = K, L = L, bayes_S_M = bayes_S_M, bayes_S_V = bayes_S_V, bayes_W_M = bayes_W_M, bayes_W_V = bayes_W_V)
np.savez('mnist_100_train_20_layers_K_250_beta_est', D = D, K = K, L = L, freq_beta_train_est = freq_beta_train_est, freq_beta_validation_est = freq_beta_validation_est, shared_beta_train_est = shared_beta_train_est, shared_beta_validation_est = shared_beta_validation_est, true_beta_train = true_beta_train, true_beta_validation = true_beta_validation, y_train=y_train, y_validation=y_validation)
np.savez('mnist_100_train_20_layers_K_250_quality', D = D, K = K, L = L, freq_train_f_measure = freq_train_f_measure, freq_train_loss = freq_train_loss, freq_validation_f_measure = freq_validation_f_measure, freq_validation_loss = freq_validation_loss, shared_bayesian_train_f_measure = shared_bayesian_train_f_measure, shared_bayesian_train_loss = shared_bayesian_train_loss, shared_bayesian_validation_f_measure = shared_bayesian_validation_f_measure, shared_bayesian_validation_loss = shared_bayesian_validation_loss)
np.savez('mnist_100_train_20_layers_K_250_params', D = D, K = K, L = L, train_size = train_size)