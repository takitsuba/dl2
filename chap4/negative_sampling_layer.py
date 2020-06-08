import sys

sys.append("..")
from common.layers import Embedding
import numpy as np
from collections import Counter


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target = self.cache

        # doutはベクトルなので、shapeを行列にしているということかな
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * dtarget_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # Counterに突っ込めば各要素の出現回数が辞書で得られるはず
        counts = Counter(corpus)
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        corpus_size = len(corpus)

        for i in range(vocab_size):
            # ある単語の確率は、ある単語の出現回数を全単語の出現回数で割り求める
            self.word_p[i] = counts[i] / corpus_size

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
