import tensorflow as tf

from tf_probability.spike_and_slab_model import spike_and_slab_model


class SpikeAndSlabModelTest(tf.test.TestCase):
    def sample_from_model(self):
        num_datapoints = 5000
        data_dim = 2
        latent_dim = 1
        stddv_datapoints = 0.5

        model = spike_and_slab_model(data_dim=data_dim,
                                  latent_dim=latent_dim,
                                  num_datapoints=num_datapoints,
                                  stddv_datapoints=stddv_datapoints)

        with tf.Session() as sess:
            x_train, (actual_w, actual_z) = sess.run(model)

        print("Principal axes:")
        print(actual_w)

        self.assertAllEqual(x_train.shape, (2, 3))

    def sample_from_model2(self):
        num_datapoints = 5000
        data_dim = 2
        latent_dim = 1
        stddv_datapoints = 0.5

        model = spike_and_slab_model(data_dim=data_dim,
                                  latent_dim=latent_dim,
                                  num_datapoints=num_datapoints,
                                  stddv_datapoints=stddv_datapoints)

        with tf.Session() as sess:
            x_train, (actual_w, actual_z) = sess.run(model)

        print("Principal axes:")
        print(actual_w)

        self.assertAllEqual(x_train.shape, (2, 3))

if __name__ == '__main__':
    tf.test.main()