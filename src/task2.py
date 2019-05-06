import numpy as np
from optimizers import gradient_decent
from preprocess import X0_initialize


def init_graph_task2():
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


def training(graph_matrix, n_vector, y, 
            W, A, b, alpha=0.0001, eps=0.001, gnn_steps=2, n_itr=200):
    """与えられたgraph_matrix, 正解ラベルyに対してgradient decentを実行する"""

    for itr in range(n_itr):
        W, A, b, L = gradient_decent(graph_matrix,n_vector,y,W,A,b,gnn_steps,alpha,eps,out_flag='val')
        print('iteration: {}, loss: {}'.format(itr+1, L))
    return L


if __name__ == '__main__':

    # graph_matrixの初期化
    graph_matrix = init_graph_task2()

    # グラフのノード数、特徴量ベクトルの次元
    n_graph = graph_matrix.shape[0]
    n_vector = 8

    # パラメータの初期化
    np.random.seed(3)
    W = np.random.normal(0, 0.4, (n_vector, n_vector))
    np.random.seed(3)
    A = np.random.normal(0, 0.4, n_vector)
    b = 0

    # 訓練(gradient decent)の実行
    n_itr = 200
    loss = training(graph_matrix=graph_matrix, n_vector=n_vector, y=1, W=W, A=A, b=b, n_itr=n_itr)

    # 最終的なloss
    print('itr: {}, loss: {}'.format(n_itr, loss))