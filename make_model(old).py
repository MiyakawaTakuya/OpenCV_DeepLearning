# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from chainer import Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_openml

#学習モデル(多層のニューラルネットワーク)の設定
class MyMLP(Chain):
    #__init__メソッドで学習モデルを作成
    def __init__(self, n_in=784, n_units=100, n_out=10):
        super(MyMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )
    #__call__メソッドで先ほど作成した学習モデルとF/reru関数をしようし入力xを出力y
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
    
#プログラムが開始したことを示す(MNISTのダウンロードに時間を要するため)
print('Start')

#データセットの準備
mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

#データセットの変換 MNISTのデータセットに格納する。データ型をfloat64からそれぞれ関数を使って変換している
x_all = mnist_X.astype(np.float32) / 255  #mnist_Xに画像が格納されている
y_all = mnist_y.astype(np.int64)  #mnist_yに数字のラベルが格納されている

#optimizerの作成
model = MyMLP()
optimizer = optimizers.SGD()  #確率的勾配降下法と呼ばれるもっとも簡単なアルゴリズムであるコンストラクタを使用
optimizer.setup(model)   #optimizer.setup()関数で引数に与えた学習モデルを最適化する

#optimizerの最適化
BATCHSIZE = 100
DATASIZE = 70000

for epoch in range(20):
    print('epoch %d' % epoch)
    indexes = np.random.permutation(DATASIZE)
    for i in range(0, DATASIZE, BATCHSIZE):
        x = Variable(x_all[indexes[i : i + BATCHSIZE]])
        t = Variable(y_all[indexes[i : i + BATCHSIZE]])
        
        model.zerograds() #前回のループで計算された勾配を０に初期化している
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)  #正解ラベルと予測との損失値(？？)
        loss.backward()  #勾配を計算 逆伝播計算
        optimizer.update()  #逆伝播計算した結果を元に最適化を行う

#保存 npzファイルで保存
serializers.save_npz("mymodel.npz", model)

#プログラムが終了したことを示す
print('Finish')




