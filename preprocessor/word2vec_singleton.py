# word2vec_singleton.py
import os
import gensim

class Word2VecSingleton:
    _instance = None
    model = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Word2VecSingleton()
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        bin_path = "flask-server\data\word2vec\GoogleNews-vectors-negative300.bin"
        kv_path = os.path.join("data", "word2vec", "word2vec_prepared.kv")
        
        # Load from the optimized file if it exists
        if os.path.exists(kv_path):
            print(f"Loading optimized model from: {kv_path}")
            self.model = gensim.models.KeyedVectors.load(kv_path)
        elif os.path.exists(bin_path):
            # Else, load from binary and save for next time
            print(f"Loading raw model from: {bin_path}")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(bin_path, binary=True, limit=1000000)
            print("Pre-normalizing vectors...")
            self.model.fill_norms()

            # Ensure save directory exists
            os.makedirs(os.path.dirname(kv_path), exist_ok=True)

            self.model.save(kv_path)
            print(f"Saved optimized model to: {kv_path}")
        else:
            print(f"Model file not found at {bin_path}")
            self.model = None
    
    def get_model(self):
        return self.model

def get_word2vec_model():
    return Word2VecSingleton.get_instance().get_model()