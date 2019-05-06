import numpy as np
from networks import GNN_agg_readout
from preprocess import X0_initialize
from functions import sigmoid
from optimizers import loss_function


def mean_precision(data, W, A, b, n_vector, gnn_steps):
    """平均精度を計算する"""
    # 0, 1と予測された場合のcount
    judge0 = 0
    judge1 = 0
    # 予測されたlabelと真のlabelがいずれも0, 1だった場合のカウント
    same_num0 = 0
    same_num1 = 0
    for d in data:
        h = GNN_agg_readout(d[0], n_vector, W, gnn_steps)
        s = np.dot(A, h)+b
        y = sigmoid(s)
        if y < 0.5:
            judge0 += 1
            if d[1] == 0:
                same_num0 += 1
        else:
            judge1 += 1
            if d[1] == 1:
                same_num1 += 1
    assert same_num0 <= judge0, "precision calculation is not correct!"
    assert same_num1 <= judge1, "precision calculation is not correct!"
    delta = 0.00000001
    precision0 = same_num0 / (judge0+delta)
    precision1 = same_num1 / (judge1+delta)
    mean_precision = (precision0 + precision1)/2
    return mean_precision


def valid_loss(data, W, A, b, n_vector, gnn_steps):
    """検証データのlossを計算する"""
    n_data = len(data)
    loss_arr = np.zeros(n_data)
    for i, d in enumerate(data):
        loss = loss_function(d[0], n_vector, d[1], W, A, b, gnn_steps)
        loss_arr[i] = loss
    # lossの平均をとる
    mean_loss = np.mean(loss_arr)
    return mean_loss

        