import tensorflow as tf
from constants import *
from networks.layers import *

#import tensorflow_probability as tfp


class BetweenClusterFC(tf.keras.Model):

    def __init__(self, n_clusters):
        super(BetweenClusterFC, self).__init__()
        
        self.fcs = {}

        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                self.fcs[(i,j)] = (tf.keras.layers.Dense(
                                FC_OUTPUT_DIMENSION, activation=lambda x : x
                            ), tf.keras.layers.Dense(
                                FC_OUTPUT_DIMENSION, activation=lambda x : x
                            ))


    def call(self, emb_1, emb_2, cluster_1, cluster_2):
        fc_1, fc_2 = self.fcs[(cluster_1, cluster_2)]
        emb_1, emb_2 = fc_1(emb_1), fc_2(emb_2)

        return tf.linalg.diag_part(tf.matmul(emb_1, emb_2, transpose_b=True))

        