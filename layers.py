import tensorflow as tf


# noise_shape is [num_nonzero_elements], namely it is a list containing the number of elements of the sparse tensor
# keep_prob is 1-dropout_rate 
# inputs is the sparse tensor which we are applying dropout to
def dropout_sparse(inputs, keep_prob, noise_shape):
    keep_tensor = keep_prob + tf.random.uniform(noise_shape)
    to_retain = tf.cast(tf.floor(keep_tensor), dtype=tf.bool)
    out = tf.sparse.retain(inputs, to_retain=to_retain)

    # the elements of the tensor are rescaled after dropout
    return out * (1/keep_prob)

class GraphSparseConvolution(tf.keras.layers.Layer):
    
    def __init__(self, adj_norm, output_size=32, dropout_rate=0.0, act=tf.nn.relu):
        super(GraphSparseConvolution, self).__init__()
        self.adj_norm = adj_norm
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.act = act


    # input_shape here will be automatically set as the shape of the input tensor, that will be the feature matrix
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.kernel = self.add_weight('kernel', initializer=init, shape=[int(input_shape[-1]),self.output_size])

    # the input is a sparse tensor whose elements have been explicitly converted to floats
    # compute ReLU(Ã inputs Weights)
    def call(self, inputs, training):
        x = inputs
        if(training):
            x = dropout_sparse(inputs, 1-self.dropout_rate, [len(inputs.values)])
        x = tf.sparse.sparse_dense_matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        outputs = self.act(x)
        return outputs

# the only difference between these two classes is that the first will treat also the features as a sparse matrix (the adjacency matrix will always be sparse)
class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, adj_norm, output_size=16, dropout_rate=0.0, act=tf.nn.relu):
        super(GraphConvolution, self).__init__()
        self.adj_norm = adj_norm
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.act= act
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    
    def build(self, input_shape):
        init = tf.keras.initializers.GlorotNormal(seed=None)
        self.kernel = self.add_weight('kernel', initializer=init, shape=[int(input_shape[-1]),self.output_size])


    # the input to the call function is a dense tensor whose elements have been explicitly converted to floats
    def call(self, inputs, training):
        x = self.dropout(inputs, training=training)
        x = tf.matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        outputs = self.act(x)
        return outputs


class TopologyDecoder(tf.keras.layers.Layer):
    
    def __init__(self, act=tf.nn.sigmoid, dropout_rate=0.0):
        super(TopologyDecoder, self).__init__()
        self.act = act
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    # compute inputs inputs^T    
    def call(self, inputs, training):
        x = self.dropout(inputs, training=training)
        x = tf.matmul(x, x, transpose_b = True)
        return self.act(x)