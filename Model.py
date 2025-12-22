import tensorflow as tf
import numpy as np


class GraphConvLayer(tf.keras.layers.Layer):
    """Minimal GCN layer with symmetric normalization."""

    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        feature_shape = input_shape[0]
        input_dim = int(feature_shape[-1])
        self.weight = self.add_weight(
            name="weight",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, edge_index = inputs
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        edge_index = tf.cast(edge_index, tf.int64)
        num_nodes = tf.shape(features)[0]
        num_edges = tf.shape(edge_index)[1]
        indices = tf.transpose(edge_index)
        values = tf.ones(num_edges, dtype=tf.float32)
        dense_shape = tf.cast(tf.stack([num_nodes, num_nodes]), tf.int64)
        adj = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
        adj = tf.sparse.reorder(adj)

        degrees = tf.sparse.reduce_sum(adj, axis=1)
        degrees = tf.reshape(degrees, (-1,))
        deg_inv_sqrt = tf.pow(degrees, -0.5)
        deg_inv_sqrt = tf.where(tf.math.is_finite(deg_inv_sqrt), deg_inv_sqrt, tf.zeros_like(deg_inv_sqrt))
        row = adj.indices[:, 0]
        col = adj.indices[:, 1]
        scaled_values = adj.values * tf.gather(deg_inv_sqrt, row) * tf.gather(deg_inv_sqrt, col)
        norm_adj = tf.sparse.SparseTensor(indices=adj.indices, values=scaled_values, dense_shape=adj.dense_shape)
        norm_adj = tf.sparse.reorder(norm_adj)

        propagated = tf.sparse.sparse_dense_matmul(norm_adj, features)
        h = tf.matmul(propagated, self.weight)
        if self.bias is not None:
            h = h + self.bias
        if self.activation is not None:
            h = self.activation(h)
        return h

class TiReMGE(tf.keras.Model):
    def __init__(self, node_num, source_num, class_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcn10 = GraphConvLayer(256, activation=tf.nn.relu)

        self.gcn20 = GraphConvLayer(64, activation=None)
        self.gcn21 = GraphConvLayer(256, activation=tf.nn.relu)

        self.fc1 = tf.keras.layers.Dense(class_num)

    def call(self, input, training=None):
        x, edge_index1, edge_index2 = input

        h1 = self.gcn10([x, edge_index1])

        h2 = self.gcn20([x, edge_index2])
        h2 = self.gcn21([h2, edge_index2])

        h = h1 + h2

        h = self.fc1(h)

        return h

# import tf_geometric as tfg
# import tensorflow as tf
# import numpy as np

# class TiReMGE(tf.keras.Model):
#     def __init__(self, node_num, source_num, class_num, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.gcn10 = tfg.layers.GCN(256, activation=tf.nn.relu)

#         self.gcn20 = tfg.layers.GCN(64, activation=None)
#         self.gcn21 = tfg.layers.GCN(256, activation=tf.nn.relu)

#         self.fc1 = tf.keras.layers.Dense(class_num)

#     def call(self, input, training=None, return_hidden=False):
#         x, edge_index1, edge_index2 = input

#         h1 = self.gcn10([x, edge_index1])

#         h2 = self.gcn20([x, edge_index2])
#         h2 = self.gcn21([h2, edge_index2])

#         h = h1 + h2

#         logits = self.fc1(h)

#         if return_hidden:
#             return logits, h
#         return logits
