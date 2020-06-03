import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 上記のfor文の中で同時に作成していくことも可能で
    # そちらの方が計算回数は減りそうだが、見通しが悪そう
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        # 注目する単語の隣（i=1）からウインドウサイズの最大の遠さの
        # 単語までを数えていく
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y):
    # xとyの正規化を行ってから内積を計算する
    nx = x / np.sqrt(np.sum(x ** 2))
    ny = y / np.sqrt(np.sum(y ** 2))
    return np.dot(nx, ny)
