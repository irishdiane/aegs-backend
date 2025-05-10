import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocessor.seed_words import seed_words_formal, seed_words_informal

class LanguageToneEvaluator:
    
    def __init__(self, word2vec_model):
        self.model = word2vec_model
        # Load necessary NLTK data
        self._ensure_nltk_resources()
        
        # Generate dynamic formal and informal word lists
        self.formal_words = self._get_formal_words()
        self.informal_words = self._get_informal_words()
    
    def _ensure_nltk_resources(self):
        resources = [
            ("corpora/brown", "brown"),
            ("tokenizers/punkt", "punkt"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")
        ]
        
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(resource_name)
    
    def _get_formal_words(self):
        similar_words = set(seed_words_formal)
        for word in seed_words_formal:
            if word in self.model.key_to_index:
                similar_words.update([w for w, _ in self.model.most_similar(word, topn=100)])
        return similar_words
    
    def _get_informal_words(self):
        similar_words = set(seed_words_informal)
        for word in seed_words_informal:
            if word in self.model.key_to_index:
                similar_words.update([w for w, _ in self.model.most_similar(word, topn=100)])
        return similar_words
    
    def get_brown_match_score(self, essay_text, category="learned"):
        brown_words = brown.words(categories=category)
        brown_freq = Counter(word.lower() for word in brown_words)
        essay_words = word_tokenize(essay_text.lower())
        match_count = sum(1 for word in essay_words if word in brown_freq)
        return match_count / len(essay_words) if essay_words else 0
    
    def get_formality_score(self, essay_text):
        words = word_tokenize(essay_text.lower())
        total = len(words)
        if total == 0:
            return 0, 0
        formal_count = sum(1 for w in words if w in self.formal_words)
        informal_count = sum(1 for w in words if w in self.informal_words)
        return formal_count / total, informal_count / total
    
    def get_pos_distribution(self, essay_text):
        tags = pos_tag(word_tokenize(essay_text))
        if not tags:
            return 0, 0
        noun_ratio = sum(1 for _, t in tags if t.startswith("NN")) / len(tags)
        pronoun_ratio = sum(1 for _, t in tags if t.startswith("PRP")) / len(tags)
        return noun_ratio, pronoun_ratio
    
    def evaluate(self, essay_text):
        brown_score = self.get_brown_match_score(essay_text)
        formal_score, informal_score = self.get_formality_score(essay_text)
        noun_ratio, pronoun_ratio = self.get_pos_distribution(essay_text)
        
        # Invert informal score and pronoun ratio since higher = more casual
        adjusted_informal_score = 1 - informal_score
        adjusted_pronoun_ratio = 1 - pronoun_ratio
        
        # Weighted average or simple average of all adjusted metrics
        scores = [
            brown_score,
            formal_score,
            adjusted_informal_score,
            noun_ratio,
            adjusted_pronoun_ratio
        ]
        
        final_score = sum(scores) / len(scores) if scores else 0
        return round(final_score, 4)