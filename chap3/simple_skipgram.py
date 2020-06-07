import sys

sys.path.append("..")
from common.layers import MatMul, SoftmaxWithLoss
import numpy as np


class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):

        V, H = vocab_size, hidden_size

        # 重みの初期設定
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        # 各レイヤを作る。
        self.in_layer = MatMul(W_in)
        # 予測すべきcontextの単語数分だけloss_layerを作成する必要がある
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        # 全てのlayer,重み,勾配をリストにまとめる
        layers = [
            self.in_layer,
            self.out_layer,
            self.loss_layer0,
            self.loss_layer1,
        ]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        score = self.out_layer.forward(h)
        loss0 = self.loss_layer0.forward(score, contexts[:, 0])
        loss1 = self.loss_layer1.forward(score, contexts[:, 1])
        loss = loss0 + loss1
        return loss

    def backward(self, dout=1):
        dl0 = self.loss_layer0.backward(dout)
        dl1 = self.loss_layer1.backward(dout)
        ds = dl0 + dl1
        da = self.out_layer.backward(ds)
        self.in_layer.backward(da)

        return None
