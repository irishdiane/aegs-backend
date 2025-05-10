import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from preprocessor.word_expander import load_expanded_vocab

stop_words = set(stopwords.words("english"))
expanded_vocab = load_expanded_vocab()

class VocabularyEvaluator:
    def __init__(self, word2vec_model):
        self.model = word2vec_model
        
    def evaluate(self, essay_text):
        #Main method to evaluate vocabulary richness.
        return self.check_vocabulary(essay_text)

    def preprocess_text(self, text):
        #Tokenizes and cleans text input.
        words = word_tokenize(text.lower())
        return [w for w in words if w.isalpha() and w not in stop_words and w in self.model.key_to_index]

    def get_weighted_average_vector(self, words):
        #Returns the weighted average Word2Vec vector of the input words.
        word_freq = Counter(words)
        total_freq = sum(word_freq.values())
        weighted_vectors = []

        for word in word_freq:
            if word in self.model.key_to_index:
                weight = word_freq[word] / total_freq
                weighted_vectors.append(self.model[word] * weight)

        if weighted_vectors:
            return np.mean(weighted_vectors, axis=0)
        return None

    def fine_grained_similarity(self, essay_vectors, vocab_vectors):
        # Computes cosine similarity between essay words and advanced vocab.
        similarities = cosine_similarity(essay_vectors, vocab_vectors)
        max_similarities = np.max(similarities, axis=1)  # highest similarity per essay word
        return round(float(np.mean(max_similarities)), 5)

    def check_vocabulary(self, essay_text):
        # Calculates vocabulary richness score based on advanced vocabulary usage.
        essay_words = self.preprocess_text(essay_text)
        if not essay_words:
            return 0

        essay_vectors = [self.model[word] for word in essay_words]
        vocab_vectors = [self.model[word] for word in expanded_vocab if word in self.model.key_to_index]

        if not vocab_vectors:
            return 0

        return self.fine_grained_similarity(essay_vectors, vocab_vectors)