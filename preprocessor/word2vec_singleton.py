import os
import gensim
import gdown

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
        model_dir = "flask-server/data/word2vec"
        kv_path = os.path.join(model_dir, "word2vec_prepared.kv")
        npy_path = kv_path + ".vectors.npy"

        os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(kv_path) or not os.path.exists(npy_path):
            print("Downloading .kv and .npy model files from Google Drive...")

            gdown.download(f"https://drive.google.com/uc?id=1rDEb-O982wpqrLswUh5CJvZD7fwnYuuD", kv_path, quiet=False)
            gdown.download(f"https://drive.google.com/uc?id=1m85AAmRRnoefyYQlyHMSGXd0bE1BI7fy", npy_path, quiet=False)

            print("Download complete.")

        print(f"Loading model from: {kv_path}")
        self.model = gensim.models.KeyedVectors.load(kv_path)

    def get_model(self):
        return self.model

def get_word2vec_model():
    return Word2VecSingleton.get_instance().get_model()
