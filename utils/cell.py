import numpy as np
import pandas as pd
#from layer_assist import Unit
#from Unit import call
import tensorflow as tf

# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def load_features(feat_path, dtype=np.float32):
    # feat_df = pd.read_csv(feat_path)
    feat_df = np.load(feat_path)
    feat = feat_df[:, :, 9:]
    # feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    # adj_df = pd.read_csv(adj_path, header=None)
    adj_df = np.load(adj_path)
    adj_df = adj_df[:, :, 6:9]
    # adj = np.array(adj_df, dtype=dtype)
    x, y, z = np.shape(adj_df)
    adj = []
    for i in range(x):
        l = []
        for j in range(y):
            item = adj_df[i][j]
            l.append(np.array([[0, item[0], item[1]], [item[0], 0, item[2]], [item[1], item[2], 0]]))
        adj.append(np.array(l))
    return adj