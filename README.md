# README

![gnn.pdf](https://github.com/Takaaki-Saeki/GNN/files/3358312/2019-07-04.17.06.41.pdf)

## 概要
某のコーディング課題を解きました。

## 環境・モジュール
pythonを用いて実装を行いました。  
実行の際は以下の環境(動作確認済み)を推奨します。
- python 3.7.3
- numpy 1.16.3
- matplotlib 3.0.3

## 提出物について
1. `src`
2. `README.md`
3. `prediction.txt`
4. `report.pdf`
5. `results`  
なお、`results`には学習済みのlossや重みの値が入っています。

## 実行方法
以下の手順で実行してください。  
1. `src`ディレクトリと`datasets`ディレクトリを同じディレクトリ内に置いてください。  
2. `src`ディレクトリと`datasets`ディレクトリのあるディレクトリに、`results`という名前のディレクトリを作成してください。学習を回さずに計算済みの重みやlossの値を使う場合は、提出した`results`ディレクトリを使ってください。学習を回す場合は`results`ディレクトリ内を空にしてください。
3. `src`ディレクトリ内に課題1〜4のスクリプト`task1.py`〜`task4.py`があります。
   `src`ディレクトリに移動し、以下のコマンドから実行してください。  
   
   ```
   python -u task1.py
   python -u task2.py
   python -u task3.py
   python -u task4.py
   ```
   なお、current directoryが`src`になっていない場合は、Assertion Errorが出るようになっ    ています。


## ソースコードの構成について
ファイル・関数構成は以下のようになっています。  
詳細な説明についてはソースコード内に記してあるので、そちらを参照してください。

```
src
├─ __pycache__
|
├─ functions.py
|  ├─ relu()
|  ├─ sigmoid()
│  └─ binary_cross_entropy()
|
├─ optimizers.py
|  ├─ loss_function()
|  ├─ gradient_decent()
|  ├─ SGD()
|  ├─ Momentum_SGD()
│  └─ Adam()
|
├─ preprocess.py
|  ├─ X0_initialize()
│  └─ train_valid_split()
|
├─ networks.py
│  └─ GNN_agg_readout()
|
├─ evaluation.py
|  ├─ mean_precision()
│  └─ valid_loss()
|
├─ predict.py
|  └─ predict()
|
├─ in_out_data.py
|  ├─ get_train_data()
|  ├─ get_test_data()
|  └─ out_test_label()
|
├─ visualize.py
|  ├─ plot_loss_precision_task3()
|  └─ plot_loss_precision_task4()
|
├─ task1.py
|  ├─ init_graph_task1()
|  ├─ test1()
│  └─ test2()
|
├─ task2.py
│  └─ training()
|
├─ task3.py
|  ├─ run_GNN_SGD()
│  └─ run_GNN_MomentumSGD()
|
└─ task4.py
   └─ run_GNN_Adam()
```



