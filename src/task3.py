import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from optimizers import SGD
from optimizers import Momentum_SGD
from preprocess import train_valid_split
from evaluation import mean_precision
from evaluation import valid_loss
from visualize import plot_loss_precision_task3
from in_out_data import get_train_data


def run_GNN_SGD(train_data, valid_data, W, A, b, B, alpha=0.0001,
            eps=0.001, n_vector=8, gnn_steps=2, n_epochs=100):
    params = []
    for epoch in range(n_epochs):
        W, A, b, loss_train = SGD(train_data, n_vector, B, W, A, b, gnn_steps, alpha, eps)
        precision_train = mean_precision(train_data, W, A, b, n_vector, gnn_steps)
        precision_val = mean_precision(valid_data, W, A, b, n_vector, gnn_steps)
        loss_val = valid_loss(data, W, A, b, n_vector, gnn_steps)
        print('epoch: {}, train loss: {}, train precision: {}, valid loss: {}, valid precision: {}'.format(epoch+1, loss_train, precision_train, loss_val, precision_val))
        params.append((loss_train, precision_train, loss_val, precision_val))
    return params


def run_GNN_MomentumSGD(train_data, valid_data, W, A, b, B, alpha=0.0001, eta=0.9, 
            eps=0.001, n_vector=8, gnn_steps=2, n_epochs=100):
    w_W = np.zeros(W.shape)
    w_A = np.zeros(A.shape)
    w_b = 0
    params = []
    for epoch in range(n_epochs):
        W, A, b, loss_train = Momentum_SGD(train_data, n_vector, B, W, A, b, gnn_steps, alpha, eta, eps, w_W, w_A, w_b)
        precision_train = mean_precision(train_data, W, A, b, n_vector, gnn_steps)
        precision_val = mean_precision(valid_data, W, A, b, n_vector, gnn_steps)
        loss_val = valid_loss(data, W, A, b, n_vector, gnn_steps)
        print('epoch: {}, train loss: {}, train precision: {}, valid loss: {}, valid precision: {}'.format(epoch+1, loss_train, precision_train, loss_val, precision_val))
        params.append((loss_train, precision_train, loss_val, precision_val))
    return params


if __name__ == '__main__':

    # trainディレクトリ内のファイルを取得する
    data = get_train_data()

    # train : valid = 7 : 3 の比率で分割する
    train_data, valid_data = train_valid_split(data, 0.3)

    # パラメータの初期化
    n_vector = 8

    # エポック数, バッチサイズの設定
    n_epochs = 100
    B = 100

    # SGDで、計算済みのparamsが存在するならそれを使い、存在しないなら学習を回す
    if os.path.exists('../results/params_SGD_{}.npy'.format(B)):
        params_SGD = np.load('../results/params_SGD_{}.npy'.format(B), allow_pickle = True)
    else:
        # SGDのパラメータの初期化
        np.random.seed(42)
        W = np.random.normal(0, 0.4, (n_vector, n_vector))
        np.random.seed(42)
        A = np.random.normal(0, 0.4, n_vector)
        b = 0
        # SGDの訓練を行い、平均loss, 平均精度を求める
        params_SGD = run_GNN_SGD(train_data=train_data, valid_data=valid_data, W=W, A=A, b=b, B=B, n_vector=n_vector, n_epochs=n_epochs)
        np.save('../results/params_SGD_{}.npy'.format(B), params_SGD)
    
    #MomentumSGDで、計算済みのparamsが存在するならそれを使い、存在しないなら学習を回す
    if os.path.exists('../results/params_MomentumSGD_{}.npy'.format(B)):
        params_MomentumSGD = np.load('../results/params_MomentumSGD_{}.npy'.format(B), allow_pickle = True)
    else:
        # MomentumSGDのパラメータの初期化
        np.random.seed(42)
        W = np.random.normal(0, 0.4, (n_vector, n_vector))
        np.random.seed(42)
        A = np.random.normal(0, 0.4, n_vector)
        b = 0
        # MomentumSGDの訓練を行い、平均loss, 平均精度を求める
        params_MomentumSGD = run_GNN_MomentumSGD(train_data=train_data, valid_data=valid_data, W=W, A=A, b=b, B=B, n_vector=n_vector, n_epochs=n_epochs)
        np.save('../results/params_MomentumSGD_{}.npy'.format(B), params_MomentumSGD)

    # 可視化を行う
    plot_loss_precision_task3(n_epochs, params_SGD, params_MomentumSGD, B)


