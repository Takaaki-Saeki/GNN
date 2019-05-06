import numpy as np
import glob
import os

def get_train_data():
    """train dataを取得し、(graph_matrix, label)を各要素とするlistとして保存する"""

    # train_dataを(graph_matrix, label)の形で入れるlist
    train_data = []

    # train dataの個数をカウント
    current_path = os.getcwd()
    graph_path_re = "../datasets/train/*_graph.txt"
    graph_path_list = glob.glob(os.path.join(current_path, graph_path_re))
    n_data = len(graph_path_list)
    assert n_data > 0, 'No input data! Check input path!'

    # 0~n_data-1のgraphデータ, labelデータを読み取り、タプルの形でtrain_dataにappendしていく
    for n in range(n_data):
        graph_path = os.path.join(current_path, "../datasets/train/{}_graph.txt".format(n))
        label_path = os.path.join(current_path, "../datasets/train/{}_label.txt".format(n))
        f_graph = open(graph_path)
        f_label = open(label_path)
        n_graph = int(f_graph.readline())
        G = np.zeros((n_graph, n_graph))
        label = int(f_label.read())
        for i in range(n_graph):
            line = f_graph.readline()
            line_list = line.split()
            for j in range(n_graph):
                G[i][j] = line_list[j]
        train_data.append((G, label))
    return train_data


def get_test_data():
    """test dataを取得し、graph_matrixを各要素とするlistとして保存する"""

    # graph matrixを格納するlist
    test_data = []

    # train dataの個数をカウント
    current_path = os.getcwd()
    graph_path_re = "../datasets/test/*_graph.txt"
    graph_path_list = glob.glob(os.path.join(current_path, graph_path_re))
    n_data = len(graph_path_list)
    # 入力がうまくいってない場合はassertを出す
    assert n_data > 0, 'No input data! Check input path!'

    # 0~n_data-1のgraphデータ, labelデータを読み取り、タプルの形でtrain_dataにappendしていく
    for n in range(n_data):
        graph_path = os.path.join(current_path, "../datasets/test/{}_graph.txt".format(n))
        f_graph = open(graph_path)
        n_graph = int(f_graph.readline())
        G = np.zeros((n_graph, n_graph))
        for i in range(n_graph):
            line = f_graph.readline()
            line_list = line.split()
            for j in range(n_graph):
                G[i][j] = line_list[j]
        test_data.append(G)
    return test_data


def out_test_label(label, file_path):
    """labelをファイルに書き込む"""
    path_w = file_path

    with open(path_w, mode='w') as f:
        for l in label:
            f.write('{}\n'.format(l))


