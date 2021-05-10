import tensorflow as tf
from constants import *
import tensorflow_probability as tfp

# loss that considers the error reconstructing some edges
def topological_loss(y_actual, y_pred):
    #return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_actual, y_pred)

    loss =  tf.nn.weighted_cross_entropy_with_logits(
        y_actual, y_pred, POS_WIGHT
    )

    return tf.math.reduce_mean(loss)

# get the total loss as topological loss - LK div
def total_loss(y_actual, y_pred, logvar, mu, n_nodes, norm):
    top_loss = topological_loss(y_actual, y_pred)

    a = tf.math.reduce_sum(1 + 2 * logvar - tf.math.pow(mu, 2) - tf.math.pow(tf.math.exp(logvar), 2), 1)

    KLD = -0.5 / n_nodes * tf.math.reduce_mean(a)

    return norm * top_loss + KLD

