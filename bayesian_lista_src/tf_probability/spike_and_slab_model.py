import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

def spike_and_slab_model(data_dim, latent_dim, num_datapoints, stddv_datapoints): # (unmodeled) data
  w = ed.Normal(loc=tf.zeros([data_dim, latent_dim]),
                scale=2.0 * tf.ones([data_dim, latent_dim]),
                name="w")  # parameter
  z = ed.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                scale=tf.ones([latent_dim, num_datapoints]),
                name="z")  # parameter
  x = ed.Normal(loc=tf.matmul(w, z),
                scale=stddv_datapoints * tf.ones([data_dim, num_datapoints]),
                name="x")  # (modeled) data
  return x, (w, z)

def get_joint_fn():
    log_joint = ed.make_log_joint_fn(spike_and_slab_model)

    return log_joint