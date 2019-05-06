import numpy as np
from functions import relu
from networks import GNN_agg_readout
from preprocess import X0_initialize


def init_graph_task1():
    """10*10のグラフを生成"""
    gmat = np.zeros((10, 10))
    gmat[0][5] = 1
    gmat[1][6] = 1
    gmat[1][7] = 1
    gmat[2][3] = 1
    gmat[3][2] = 1
    gmat[3][9] = 1
    gmat[4][5] = 1
    gmat[5][0] = 1
    gmat[5][4] = 1
    gmat[5][8] = 1
    gmat[6][1] = 1
    gmat[7][1] = 1
    gmat[8][5] = 1
    gmat[9][3] = 1
    return gmat


def test1():
    """G, Wを単位行列に選んだ場合のtest"""
    for n_graph in [8, 10, 12, 14]:
        for n_vector in [4, 6, 8, 10, 12]:
            for T in [2, 4, 6, 8, 10]:
                gmat = np.eye(n_graph, n_graph)
                W = np.eye(n_vector, n_vector)
                h = GNN_agg_readout(gmat, n_vector, W, T)
                ans = np.zeros(n_vector)
                ans[0] = n_graph
                if (h != ans).any():
                    return 1
    return 0


def test2():
    """Gを単位行列に、Wを単位行列*(-1)に選んだ場合のtest"""
    for n_graph in [8, 10, 12, 14]:
        for n_vector in [4, 6, 8, 10, 12]:
            for T in [2, 4, 6, 8, 10]:
                gmat = np.eye(n_graph, n_graph)
                W = np.eye(n_vector, n_vector) * (-1)
                h = GNN_agg_readout(gmat, n_vector, W, T)
                ans = np.zeros(n_vector)
                if (h != ans).any():
                    return 1
    return 0


if __name__ == '__main__':

    # グラフの隣接行列の初期化
    graph_matrix = init_graph_task1()

    # グラフのノード数、特徴量ベクトルの次元
    n_graph = graph_matrix.shape[0]
    n_vector = 8

    # パラメータ行列Wを、平均0, 標準偏差0.4の正規乱数で初期化
    W = np.random.normal(0, 0.4, (n_vector, n_vector))

    # GNNの集約・readoutを行い、hを求める
    h = GNN_agg_readout(graph_matrix, n_vector, W, 2)

    print("h: {}".format(h))

    # 2種類のtestを実行
    assert test1() == 0, "Calculation is not correct!"
    assert test2() == 0, "Calculation is not correct!"