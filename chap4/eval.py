import sys

sys.path.append("..")
import pickle
from common.util import most_similar

pkl_file = (
    "/Users/hiro.takizawa/dev/book/deep-learning-from-scratch-2/ch04/cbow_params.pkl"
)

with open(pkl_file, "rb") as f:
    params = pickle.load(f)
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
