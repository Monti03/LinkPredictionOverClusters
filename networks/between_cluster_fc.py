import tensorflow as tf
from tensorflow.python.ops.gen_linalg_ops import matrix_triangular_solve
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


    def call(self, emb_1, emb_2, nodes_from_to, cluster_1, cluster_2, matrix_opetaion=False):
        fc_1, fc_2 = self.fcs[(cluster_1, cluster_2)]
        emb_1, emb_2 = fc_1(emb_1), fc_2(emb_2)

        if matrix_opetaion:
            complete_between_graph_preds = tf.matmul(emb_1, emb_2, transpose_b=True)
            return tf.gather_nd(complete_between_graph_preds, nodes_from_to[:,:2])
        else:
            embs_from_ = tf.gather(emb_1, nodes_from_to[:,0])
            embs_to_ = tf.gather(emb_2, nodes_from_to[:,1])

            return tf.linalg.diag_part(tf.matmul(embs_from_, embs_to_, transpose_b=True))

        