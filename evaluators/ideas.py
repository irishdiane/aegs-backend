import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class IdeasEvaluator:

    def __init__(self, word2vec_model):
        self.model = word2vec_model
    
    def calculate_similarity(self, essay_text, prompt):
        essay_words = [word for word in essay_text.split() if word in self.model.key_to_index]
        theme_words = [word for word in prompt if word in self.model.key_to_index]
        
        if not essay_words or not theme_words:
            return 0  # No valid words found in model
            
        essay_vectors = np.array([self.model[word] for word in essay_words])
        theme_vectors = np.array([self.model[word] for word in theme_words])
        
        # Ensure averaging doesn't cause issues
        essay_avg_vector = np.mean(essay_vectors, axis=0) if essay_vectors.size else np.zeros(self.model.vector_size)
        theme_avg_vector = np.mean(theme_vectors, axis=0) if theme_vectors.size else np.zeros(self.model.vector_size)
        
        similarity = cosine_similarity([essay_avg_vector], [theme_avg_vector])[0][0]
        return float(similarity)  # Ensure output is scalar, not array
    
    def evaluate(self, essay_text, theme):
        theme_list = theme.split()
        return self.calculate_similarity(essay_text, theme_list)