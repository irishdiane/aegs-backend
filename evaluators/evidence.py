import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor.seed_words import seed_words_evidence

class EvidenceEvaluator:    
    def __init__(self, word2vec_model):
        self.model = word2vec_model

        # Load spaCy English model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Generate dynamic reasoning words
        self.reasoning_words = self._get_reasoning_words()
    
    def _get_reasoning_words(self):
        similar_words = set(seed_words_evidence)  # Start with seed words
        for word in seed_words_evidence:
            if word in self.model.key_to_index:  # Ensure word exists in model
                similar_words.update([w for w, _ in self.model.most_similar(word, topn=100)])  # Get top 100 similar words
        return similar_words
    
    def identify_main_claim(self, essay_text):
        doc = self.nlp(essay_text)
        for sent in doc.sents:
            # Check if sentence contains a verb (to ensure it's a claim)
            if any(token.pos_ == "VERB" for token in sent):
                return sent.text
        
        # Default to first sentence if no opinionated claim is found
        return next(doc.sents).text if doc.sents else ""
    
    def extract_supporting_sentences(self, essay_text):
        doc = self.nlp(essay_text)
        support_sentences = [
            sent.text for sent in doc.sents 
            if any(word in sent.text.lower() for word in self.reasoning_words)
        ]
        
        return support_sentences
    
    def calculate_similarity(self, main_claim, support_sentences):        
        # Tokenize and filter valid words
        claim_tokens = self.nlp(main_claim)  
        claim_vectors = [self.model[token.text] for token in claim_tokens if token.text in self.model.key_to_index]
        
        if not claim_vectors:
            return 0  # No valid words
        
        # Compute main claim vector
        claim_avg_vector = np.mean(claim_vectors, axis=0)
        # Compute similarity with each support sentence
        similarities = []
        for sentence in support_sentences:
            sentence_tokens = self.nlp(sentence)
            vectors = [self.model[token.text] for token in sentence_tokens if token.text in self.model.key_to_index]
            
            if vectors:
                support_avg_vector = np.mean(vectors, axis=0)
                similarity = cosine_similarity([claim_avg_vector], [support_avg_vector])[0][0]
                
                # Penalize sentences that are too similar (possible repetition)
                if similarity > 0.85:
                    similarity *= 0.8  # Reduce weight of redundant sentences
                
                similarities.append(similarity)
        return np.mean(similarities) if similarities else 0
    
    def evaluate(self, essay_text):
        main_claim = self.identify_main_claim(essay_text)
        support_sentences = self.extract_supporting_sentences(essay_text)
        if not support_sentences:
            return 0
        similarity = self.calculate_similarity(main_claim, support_sentences)
        return similarity