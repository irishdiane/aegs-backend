import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from preprocessor.seed_words import seed_words_transition

class OrganizationEvaluator:    
    def __init__(self, word2vec_model):
        self.model = word2vec_model
        # Generate expanded transition words
        self.transition_words = self._get_transition_words()
    
    def _get_transition_words(self):
        similar_words = set(seed_words_transition)
        
        for word in seed_words_transition:
            if word in self.model.key_to_index:
                similar_words.update([w for w, _ in self.model.most_similar(word, topn=100)])
        
        return similar_words
    
    def identify_main_claim(self, essay_text):
        sentences = sent_tokenize(essay_text)  # Split essay into sentences
        for sent in sentences:
            words = word_tokenize(sent.lower())
            if any(word in self.transition_words for word in words):
                return sent  # Return first structured sentence
        return sentences[0] if sentences else ""  # Default to first sentence
    
    def measure_coherence(self, essay_text):
        sentences = sent_tokenize(essay_text)  
        if len(sentences) < 2:
            return 0
            
        similarities = []
        for i in range(len(sentences) - 1):
            sent1_words = word_tokenize(sentences[i].lower())
            sent2_words = word_tokenize(sentences[i + 1].lower())
            
            sent1_vectors = [self.model[word] for word in sent1_words if word in self.model.key_to_index]
            sent2_vectors = [self.model[word] for word in sent2_words if word in self.model.key_to_index]
            
            if sent1_vectors and sent2_vectors:
                avg_vector1 = np.mean(sent1_vectors, axis=0)
                avg_vector2 = np.mean(sent2_vectors, axis=0)
                similarity = cosine_similarity([avg_vector1], [avg_vector2])[0][0]
                similarities.append(similarity)
                
        return np.mean(similarities) if similarities else 0
    
    def evaluate(self, essay_text):
        main_claim = self.identify_main_claim(essay_text)
        coherence_score = self.measure_coherence(essay_text)
        
        # Award bonus for using transition words
        transition_bonus = 0.1 if any(word in self.transition_words 
                                     for word in word_tokenize(essay_text.lower())) else 0
        
        organization = coherence_score + transition_bonus
        
        # Clamp between 0 and 1
        return round(min(max(organization, 0), 1.0), 5)