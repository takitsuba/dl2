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
                cnt += 1
                if cnt % (total // 100) == 0:
                    print("%.1f%% done" % (100 * cnt / total))
    return M


def create_contexts_target(corpus, window_size=1):
    # 両端のwindow_size分以外はtargetになる
    target = corpus[window_size:-window_size]

    contexts = []
    # 両端のtargetにならない単語のぞきidxを取得
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        # targetのidxから最大でwindow_size分離れた単語を取得しリストに格納
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    """one-hot表現への変換

    :param corpus: 単語IDのリスト（1次元もしくは2次元のnumpy配列)
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元もしくは3次元のnumpy配列）
    """

    N = corpus.shape[0]

    # targetの場合はcorpusが1次元
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    # contextsの場合はcorpusが2次元
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, words in enumerate(corpus):
            for idx_1, word_id in enumerate(words):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
