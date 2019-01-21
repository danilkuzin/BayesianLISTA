import tensorflow as tf
from tf.algorithms.shared.soft_thresholding import soft_threshold
from tf.data.synthetic.data_generator import DataGenerator


class DataGeneratorTest(tf.test.TestCase):

    def testCreate(self):
        with self.test_session():
            data_generator = DataGenerator(D=10, K=5, sparsity=0.8, beta_scale=1, noise_scale=0.5)

if __name__ == '__main__':
    tf.test.main()
