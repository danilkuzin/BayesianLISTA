import tensorflow as tf


def soft_threshold(v, thr_lambda):
    return tf.math.sign(v) * tf.math.maximum(tf.math.abs(v) - thr_lambda, tf.zeros_like(v))
