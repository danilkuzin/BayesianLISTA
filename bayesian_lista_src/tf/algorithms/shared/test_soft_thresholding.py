import tensorflow as tf
from tf.algorithms.shared.soft_thresholding import soft_threshold


class SoftThresholdingTest(tf.test.TestCase):

    def testScalar(self):
        with self.test_session():
            v = tf.constant(2.0)
            thr_lambda = tf.constant(0.5)
            x = soft_threshold(v, thr_lambda)
            self.assertAllEqual(x.eval(), 1.5)

    def testVector(self):
        with self.test_session():
            v = tf.constant([2.0, 0.1, 0.5, -2.0])
            thr_lambda = tf.constant(0.5)
            x = soft_threshold(v, thr_lambda)
            self.assertAllEqual(x.eval(), [1.5, 0.0, 0.0, -1.5])


if __name__ == '__main__':
    tf.test.main()
