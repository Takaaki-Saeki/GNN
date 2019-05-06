import numpy as np
from functions import relu
from preprocess import X0_initialize


def GNN_agg_readout(graph_matrix, n_vector, W, n_steps):
    """graph_matrixが与えられた時に(集約1),(集約2),(READOUT)を計算してhを返す"""
    n_graph = graph_matrix.shape[0]
    X = X0_initialize(n_vector, n_graph)
    # n_stepsだけループを回す
    for step in range(n_steps):

        # 集約1を行う 
        A = np.dot(X, graph_matrix)
        # 集約2を行う
        X = relu(np.dot(W, A))

    # 列方向に足し合わせる
    h = np.sum(X, axis=1)
    return h