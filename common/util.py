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
