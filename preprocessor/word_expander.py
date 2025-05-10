import os
import pickle
from .word2vec_singleton import get_word2vec_model
from .seed_words import advanced_vocab

model = get_word2vec_model()

def get_advanced_words():
    similar_words = set(advanced_vocab)

    if model is None:
        print("Warning: Word2Vec model not loaded.")
        return similar_words

    for word in advanced_vocab:
        if word in model.key_to_index:
            similar_words.update([w for w, _ in model.most_similar(word, topn=100)])

    return similar_words

def load_expanded_vocab(path="data/pretrained/expanded_vocab.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        expanded = get_advanced_words()
        with open(path, "wb") as f:
            pickle.dump(expanded, f)
        return expanded
