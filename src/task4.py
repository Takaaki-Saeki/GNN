import numpy as np
import matplotlib.pyplot as plt
import os
from task3 import get_train_data
from preprocess import train_valid_split
from optimizers import Adam
from visualize import plot_loss_precision_task4
from evaluation import mean_precision
from evaluation import valid_loss
from predict import predict
from in_out_data import get_test_data
from in_out_data import out_test_label


def run_GNN_Adam(train_data, valid_data, W, A, b, B, alpha=0.0001,
            eps=0.001, n_vector=8, gnn_steps=2, n_epochs=100):
    """GNN with Adamで学習, 評価を行う"""
    beta1 = 0.9
    beta2 = 0.999
    m_W = np.zeros(W.shape)
    m_A = np.zeros(A.shape)
    m_b = 0.0
    v_W = np.zeros(W.shape)
    v_A = np.zeros(A.shape)
    v_b = 0.0
    # loss, precisionの保存用
    params = []
    # W, A, bの保存用
    weights = []
    for epoch in range(n_epochs):
        W, A, b, loss_train = Adam(train_data, n_vector, B, W, A, b, gnn_steps, epoch, alpha, beta1, beta2, eps, m_W, m_A, m_b, v_W, v_A, v_b)
        precision_train = mean_precision(train_data, W, A, b, n_vector, gnn_steps)
        precision_val = mean_precision(valid_data, W, A, b, n_vector, gnn_steps)
        loss_val = valid_loss(data, W, A, b, n_vector, gnn_steps)
        print('epoch: {}, train loss: {}, train precision: {}, valid loss: {}, valid precision: {}'.format(epoch+1, loss_train, precision_train, loss_val, precision_val))
        params.append((loss_train, precision_train, loss_val, precision_val))
        weights.append((W, A, b))
    return params, weights


if __name__ == '__main__':

    # trainディレクトリからdataを取得する
    data = get_train_data()

    # train : valid = 7 : 3の比率で分割する
    train_data, valid_data = train_valid_split(data, 0.3)

    # パラメータの初期化
    n_vector = 8
    np.random.seed(42)
    W = np.random.normal(0, 0.4, (n_vector, n_vector))
    np.random.seed(42)
    A = np.random.normal(0, 0.4, n_vector)
    b = 0

    # エポック数の設定
    n_epochs = 100
    B = 10

    # 計算済みのparams, weightsが存在するならそれを使い、存在しないなら学習を回す
    if os.path.exists('../results/weights_Adam_{}.npy'.format(B)):
        params_Adam= np.load('../results/params_Adam_{}.npy'.format(B), allow_pickle = True)
        weights_Adam = np.load('../results/weights_Adam_{}.npy'.format(B), allow_pickle = True)
    else:
        params_Adam, weights_Adam = run_GNN_Adam(train_data=train_data, valid_data=valid_data, W=W, A=A, b=b, B=B, n_epochs=n_epochs)
        np.save('../results/params_Adam_{}.npy'.format(B), params_Adam)
        np.save('../results/weights_Adam_{}.npy'.format(B), weights_Adam)

    # lossとprecisionの変化を可視化
    plot_loss_precision_task4(n_epochs, params_Adam, 10)

    # 予測のためのweight
    W_opt = weights_Adam[23][0]
    A_opt = weights_Adam[23][1]
    b_opt = weights_Adam[23][2]

    # test labelを予測し、ファイルに書き込む
    test_data = get_test_data()
    label = predict(test_data, W_opt, A_opt, b_opt, n_vector, 2)
    out_test_label(label, '../prediction.txt')