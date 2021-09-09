import tensorflow as tf
from constants import *
from networks.layers import *


class MultipleModels(tf.keras.Model):

    def __init__(self, adj_norms):
        super(MultipleModels, self).__init__()        

        # one convolutional layer per cluster
        self.conv_1 = [GraphSparseConvolution(adj_norm=adj_norm, output_size=CONV1_OUT_SIZE, dropout_rate=DROPOUT, act=tf.nn.relu) for adj_norm in adj_norms]

        self.conv_2 = [GraphConvolution(adj_norm=adj_norm, output_size=CONV2_OUT_SIZE, dropout_rate=DROPOUT, act=lambda x: x) for adj_norm in adj_norms]
        

    def call(self, inputs, cluster, training):
        # since the first convolutiona layer is not shared, I have to take the right one
        x = self.conv_1[cluster](inputs, training)
        
        x = self.conv_2[cluster](x, training)

        return x
