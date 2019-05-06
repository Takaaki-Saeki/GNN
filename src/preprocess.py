import numpy as np

def X0_initialize(n_vector, n_graph):
    """特徴量行列を,各ベクトルの第一要素が1,それ以外が0となるよう初期化"""
    X0 = np.zeros((n_vector, n_graph))
    for i in range(n_graph):
        X0[0][i] = 1

    return X0


def train_valid_split(train_data, ratio=0.3, random_seed=42):
    """trainデータとvalidデータの分割"""
    n_data = len(train_data)
    np.random.seed(random_seed)
    random_index = np.random.permutation(range(n_data))
    n_data_train = int((1-ratio)*n_data)
    train_index = random_index[:n_data_train]
    valid_index = random_index[n_data_train:]
    train_data = np.array(train_data)
    train = list(train_data[train_index])
    valid = list(train_data[valid_index])
    assert len(train) == train_data.shape[0] * (1-ratio), "wrong split of train:valid !"
    assert len(valid) == train_data.shape[0] * ratio, "wrong split of train:valid !"
    return train, valid
