import numpy as np
from networks import GNN_agg_readout
from functions import sigmoid


def predict(test_data, W, A, b, n_vector, gnn_steps):
    """test_dataのlabelを予測"""
    predicted_label = []
    for d in test_data:
        h = GNN_agg_readout(d, n_vector, W, gnn_steps)
        s = np.dot(A, h)+b
        y = sigmoid(s)
        if y < 0.5:
            predicted_label.append(0)
        else:
            predicted_label.append(1)

    return predicted_label

    

