import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

def download_and_save_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'all-MiniLM-L6-v2')
    
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}. Skipping download.")
        return
        
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Saving model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)
    print("Model downloaded and saved successfully.")

if __name__ == "__main__":
    download_and_save_model()
