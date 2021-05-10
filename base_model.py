import tensorflow as tf
from constants import *
from layers import *

import tensorflow_probability as tfp


class MyModel(tf.keras.Model):

    def __init__(self, adj_norm_tensor):
        super(MyModel, self).__init__()
        self.adj = adj_norm_tensor
        
        # the first layer is a sparse conv layer since the input tensor is sparse
        self.conv_1 = GraphSparseConvolution(adj_norm=adj_norm_tensor, output_size=CONV1_OUT_SIZE, dropout_rate=DROPOUT, act=tf.nn.relu)
        self.conv_2 = GraphConvolution(adj_norm=adj_norm_tensor, output_size=CONV2_OUT_SIZE, dropout_rate=DROPOUT, act=lambda x: x)
        
        # the second and third conv layer share the same input
        #self.conv_mu = GraphConvolution(adj_norm=adj_norm_tensor, output_size=CONV_MU_OUT_SIZE, dropout_rate=DROPOUT, act=lambda x: x)
        #self.conv_logvar = GraphConvolution(adj_norm=adj_norm_tensor, output_size=CONV_VAR_OUT_SIZE, dropout_rate=DROPOUT, act=lambda x:x)
        
        
        # decoder
        # self.top_dec = TopologyDecoder(act=lambda x:x, dropout_rate=DROPOUT)

    def reparameterize(self, mu, logvar, training):
        if training:
            std = tf.math.exp(logvar)
            eps = tf.random.normal(std.shape)
            return tf.math.multiply(eps, std) + mu
        else:
            return mu

    def call(self, inputs, training):
        # firsts convolutions
        x = self.conv_1(inputs, training)
        
        x = self.conv_2(x, training)

        #self.mu = self.conv_mu(x, training)
        #self.logvar = self.conv_logvar(x, training) 

        #z = self.reparameterize(self.mu, self.logvar, training)

        # get the reconstruction of the adj
        # top = self.top_dec(z, training)
        # reshape to tensor of shape (n_nodes^2)
        return x#, self.mu, self.logvar
    