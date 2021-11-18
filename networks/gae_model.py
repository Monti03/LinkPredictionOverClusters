import tensorflow as tf
from constants import *
from networks.layers import *

#import tensorflow_probability as tfp


class GAEModel(tf.keras.Model):

    def __init__(self, adj_norm_tensor):
        super(GAEModel, self).__init__()
        self.adj = adj_norm_tensor
        
        # the first layer is a sparse conv layer since the input tensor is sparse
        self.conv_1 = GraphSparseConvolution(adj_norm=adj_norm_tensor, output_size=CONV1_OUT_SIZE, dropout_rate=DROPOUT, act=tf.nn.relu)
        self.conv_2 = GraphConvolution(adj_norm=adj_norm_tensor, output_size=CONV2_OUT_SIZE, dropout_rate=DROPOUT, act=lambda x: x)
        

    def call(self, inputs, training):
        # firsts convolutions
        x = self.conv_1(inputs, training)
        
        x = self.conv_2(x, training)

        return x#, self.mu, self.logvar
    