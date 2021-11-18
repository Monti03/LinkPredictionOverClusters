import tensorflow as tf
from constants import *


# loss that considers the error reconstructing some edges
def topological_loss(y_actual, y_pred, sample_weight=None):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(y_actual, y_pred, sample_weight=sample_weight)

   
# get the total loss as topological loss - LK div
def total_loss(y_actual, y_pred, logvar, mu, n_nodes, norm):
    top_loss = topological_loss(y_actual, y_pred)

    a = tf.math.reduce_sum(1 + 2 * logvar - tf.math.pow(mu, 2) - tf.math.pow(tf.math.exp(logvar), 2), 1)

    KLD = -0.5 / n_nodes * tf.math.reduce_mean(a)

    return norm * top_loss + KLD

