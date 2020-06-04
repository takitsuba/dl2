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


def cos_similarity(x, y, eps=1e-8):
    # xとyの正規化を行ってから内積を計算する
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top):
    # queryがそもそもなければ何もできない
    if query not in word_to_id:
        print("%s is not found" % query)
        return

    print("\n[query]", query)
    # queryのベクトルを取り出す
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # queryのベクトルとその他のベクトルのコサイン類似度を
    vocab_size = len(word_to_id)
    similarities = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarities[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    # -1をかけているのはsimilaritiesが大きい順にしたいため
    for i in (-1 * similarities).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarities[i]))
        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # np.log2(0) = -infを避けるために微小な値epsを足している
            pmi = np.log2(C[i, j] * N / S[i] * S[j] + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                if cnt % (total // 100) == 0:
                    print("%.1f%% done" % (100 * cnt / total))
    return M
