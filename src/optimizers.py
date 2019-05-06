import numpy as np
from networks import GNN_agg_readout
from functions import binary_cross_entropy
from preprocess import X0_initialize


def loss_function(graph_matrix, n_vector, y, W, A, b, gnn_steps):
    """lossを求める"""
    h = GNN_agg_readout(graph_matrix, n_vector, W, n_steps=gnn_steps)
    s = np.dot(A, h)+b
    L = binary_cross_entropy(y, s)
    assert L != np.inf, "loss has inf value!"
    assert L != None, "loss han None value!"
    return L


def gradient_decent(graph_matrix, n_vector, y, 
                    W, A, b, gnn_steps, alpha, eps, out_flag='grad'):
    """勾配降下を行う"""
    L = loss_function(graph_matrix, n_vector, y, W, A, b, gnn_steps)
    # grad_Wを求める
    grad_W = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            W_eps = W.copy()
            W_eps[i][j] += eps
            L_W = loss_function(graph_matrix, n_vector, y, W_eps, A, b, gnn_steps)
            grad_W[i][j] = (L_W - L)/eps
    
    # grad_Aを求める
    grad_A = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        A_eps = A.copy()
        A_eps[i] += eps
        L_A = loss_function(graph_matrix, n_vector, y, W, A_eps, b, gnn_steps)
        grad_A[i] = (L_A - L)/eps
    
    # grad_bを求める
    b_eps = b + eps
    L_b = loss_function(graph_matrix, n_vector, y, W, A, b_eps, gnn_steps)
    grad_b = (L_b - L)/eps

    # パラメータの更新
    W = W - alpha*grad_W
    A = A - alpha*grad_A
    b = b - alpha*grad_b
    if out_flag=='val':
        return W, A, b, L
    else:
        return grad_W, grad_A, grad_b


def SGD(train_data, n_vector, B, W, 
        A, b, gnn_steps, alpha, eps):
    n_data = len(train_data)
    """SGDでエポックごとのパラメータ更新"""
    # 0~n_data-1の整数列をシャッフルしてindexとする
    random_index = np.random.permutation(range(n_data))
    # バッチの与え方が半端であってもsegmentation faultが起きないように, 末尾に先頭のB個を追加しておく
    random_index = np.hstack((random_index, random_index[:B]))

    # 各エポックについてn_itr_per_epoch回のiterationを行う
    n_itr_per_epoch = int(n_data/B)
    for n in range(n_itr_per_epoch):

        # 平均を算出するためのarrayを用意する
        B_data = train_data[B*n:B*(n+1)]
        grad_W_arr = np.zeros((B, W.shape[0], W.shape[1]))
        grad_A_arr = np.zeros((B, A.shape[0]))
        grad_b_arr = np.zeros(B)
        L_arr = np.zeros(B)

        # バッチサイズ個のgrad, lossをarrayに格納していく
        for i, d in enumerate(B_data):
            grad_W_d, grad_A_d, grad_b_d = gradient_decent(d[0], n_vector, d[1], W, A, b, gnn_steps, alpha, eps, out_flag='grad')
            L_d = loss_function(d[0], n_vector, d[1], W, A, b, gnn_steps)
            grad_W_arr[i] = grad_W_d
            grad_A_arr[i] = grad_A_d
            grad_b_arr[i] = grad_b_d
            L_arr[i] = L_d
        
        # 平均を計算
        grad_W = np.mean(grad_W_arr, axis=0)
        grad_A = np.mean(grad_A_arr, axis=0)
        grad_b = np.mean(grad_b_arr)
        L = np.mean(L_arr)

        # 各パラメータを更新
        W = W - alpha*grad_W
        A = A - alpha*grad_A
        b = b - alpha*grad_b

    return W, A, b, L 


def Momentum_SGD(train_data, n_vector, B, W, A, b, 
                gnn_steps, alpha, eta, eps, w_W, w_A, w_b):
    n_data = len(train_data)
    """MomentumSGDでエポックごとのパラメータ更新"""
    # 0~n_data-1の整数列をシャッフルしてindexとする
    random_index = np.random.permutation(range(n_data))
    # バッチの与え方が半端であってもsegmentation faultが起きないように, 末尾に先頭のB個を追加しておく
    random_index = np.hstack((random_index, random_index[:B]))

    # 各エポックについてn_itr_per_epoch回のiterationを行う
    n_itr_per_epoch = int(n_data/B)
    for n in range(n_itr_per_epoch):

        # 平均を算出するためのarrayを用意する
        B_data = train_data[B*n:B*(n+1)]
        grad_W_arr = np.zeros((B, W.shape[0], W.shape[1]))
        grad_A_arr = np.zeros((B, A.shape[0]))
        grad_b_arr = np.zeros(B)
        L_arr = np.zeros(B)

        # バッチサイズ個のgrad, lossをarrayに格納していく
        for i, d in enumerate(B_data):
            grad_W_d, grad_A_d, grad_b_d = gradient_decent(d[0], n_vector, d[1], W, A, b, gnn_steps, alpha, eps, out_flag='grad')
            L_d = loss_function(d[0], n_vector, d[1], W, A, b, gnn_steps)
            grad_W_arr[i] = grad_W_d
            grad_A_arr[i] = grad_A_d
            grad_b_arr[i] = grad_b_d
            L_arr[i] = L_d
        
        # 平均を計算
        grad_W = np.mean(grad_W_arr, axis=0)
        grad_A = np.mean(grad_A_arr, axis=0)
        grad_b = np.mean(grad_b_arr)
        L = np.mean(L_arr)

        # 各パラメータを更新
        W = W - alpha*grad_W+eta*w_W
        A = A - alpha*grad_A+eta*w_A
        b = b - alpha*grad_b+eta*w_b
        w_W = eta*w_W - alpha*grad_W
        w_A = eta*w_A - alpha*grad_A
        w_b = eta*w_b - alpha*grad_b

    return W, A, b, L


def Adam(train_data, n_vector, B, W, A, b, gnn_steps, epoch, alpha, 
        beta1, beta2, eps, m_W, m_A, m_b, v_W, v_A, v_b):
    """Adamでエポックごとのパラメータ更新"""
    n_data = len(train_data)
    # 0~n_data-1の整数列をシャッフルしてindexとする
    random_index = np.random.permutation(range(n_data))
    # バッチの与え方が半端であってもsegmentation faultが起きないように, 末尾に先頭のB個を追加しておく
    random_index = np.hstack((random_index, random_index[:B]))

    delta = 1e-10
    delta_W = np.full(W.shape, delta)
    delta_A = np.full(A.shape, delta)
    delta_b = delta 

    # 各エポックについてn_itr_per_epoch回のiterationを行う
    n_itr_per_epoch = int(n_data/B)
    n_itr = n_itr_per_epoch*epoch
    for n in range(n_itr_per_epoch):

        # 平均を算出するためのarrayを用意する
        B_data = train_data[B*n:B*(n+1)]
        grad_W_arr = np.zeros((B, W.shape[0], W.shape[1]))
        grad_A_arr = np.zeros((B, A.shape[0]))
        grad_b_arr = np.zeros(B)
        L_arr = np.zeros(B)

        # バッチサイズ個のgrad, lossをarrayに格納していく
        for i, d in enumerate(B_data):
            grad_W_d, grad_A_d, grad_b_d = gradient_decent(d[0], n_vector, d[1], W, A, b, gnn_steps, alpha, eps, out_flag='grad')
            L_d = loss_function(d[0], n_vector, d[1], W, A, b, gnn_steps)
            grad_W_arr[i] = grad_W_d
            grad_A_arr[i] = grad_A_d
            grad_b_arr[i] = grad_b_d
            L_arr[i] = L_d
        
        # iteration回数を補正
        n_itr += 1
        # 平均を計算
        grad_W = np.mean(grad_W_arr, axis=0)
        grad_A = np.mean(grad_A_arr, axis=0)
        grad_b = np.mean(grad_b_arr)
        L = np.mean(L_arr)

        # 各パラメータを更新
        m_W = (beta1*m_W + (1-beta1)*grad_W)/(1-beta1**n_itr)
        m_A = (beta1*m_A + (1-beta1)*grad_A)/(1-beta1**n_itr)
        m_b = (beta1*m_b + (1-beta1)*grad_b)/(1-beta1**n_itr)
        v_W = (beta2*v_W + (1-beta2)*grad_W**2)/(1-beta2**n_itr)
        v_A = (beta2*v_A + (1-beta2)*grad_A**2)/(1-beta2**n_itr)
        v_b = (beta2*v_b + (1-beta2)*grad_b**2)/(1-beta2**n_itr)
        W = W - alpha*m_W / (np.sqrt(v_W)+delta_W)
        A = A - alpha*m_A / (np.sqrt(v_A)+delta_A)
        b = b - alpha*m_b / (np.sqrt(v_b)+delta_b)

    return W, A, b, L