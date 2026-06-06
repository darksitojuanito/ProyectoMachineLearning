import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_embedding_model(model_path="models/all-MiniLM-L6-v2"):
    if os.path.exists(model_path):
        print(f"Loading local model from {model_path}...")
        model = SentenceTransformer(model_path)
    else:
        print("Local model not found. Downloading from HuggingFace...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def load_articles_dataset(path="data/DataSetArticulos.csv"):
    try:
        df = pd.read_csv(path, sep=';')
        if len(df.columns) < 2:
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path, sep=',')
        
    required_cols = ['paper_id', 'Title', 'Abstract', 'Session', 'year']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    df['Abstract'] = df['Abstract'].fillna("")
    return df

def generate_abstract_embeddings(df, model):
    abstracts = df['Abstract'].tolist()
    embeddings = model.encode(abstracts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings

def save_embeddings(embeddings, path="data/abstract_embeddings.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)

def save_embedding_metadata(df, path="data/embedding_metadata.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ['paper_id', 'Title', 'Abstract', 'Session', 'year']
    df[cols].to_csv(path, index=False)

def load_embeddings(path="data/abstract_embeddings.npy"):
    return np.load(path)

def build_embedding_similarity_matrix(embeddings):
    matrix = cosine_similarity(embeddings)
    np.fill_diagonal(matrix, 1.0)
    return matrix

def save_similarity_matrix(matrix, path="data/matriz_embeddings_coseno.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.to_csv(path, index=False)

def search_by_embeddings(query, df, model, embeddings, top_n=10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for rank, idx in enumerate(top_indices, 1):
        row = df.iloc[idx]
        results.append({
            "rank": rank,
            "paper_id": row['paper_id'],
            "Title": row['Title'],
            "Abstract": row['Abstract'],
            "Session": row['Session'],
            "year": row['year'],
            "embedding_score": similarities[idx]
        })
    
    return pd.DataFrame(results)

def recommend_by_embeddings(top10_results, df, similarity_matrix, top_k=3):
    top10_ids = set(top10_results['paper_id'].tolist())
    recommendations = []
    
    for _, row in top10_results.iterrows():
        source_id = row['paper_id']
        source_title = row['Title']
        
        # Get index of source paper in df
        source_idx = df.index[df['paper_id'] == source_id].tolist()[0]
        
        # Get similarities for this paper
        sims = similarity_matrix[source_idx]
        
        # Sort indices descending
        sorted_indices = np.argsort(sims)[::-1]
        
        recs_found = 0
        for idx in sorted_indices:
            rec_paper_id = df.iloc[idx]['paper_id']
            
            # Exclude self and top 10
            if rec_paper_id == source_id or rec_paper_id in top10_ids:
                continue
                
            rec_row = df.iloc[idx]
            recommendations.append({
                "source_paper_id": source_id,
                "source_title": source_title,
                "recommended_paper_id": rec_row['paper_id'],
                "recommended_title": rec_row['Title'],
                "recommended_session": rec_row['Session'],
                "similarity_score": sims[idx]
            })
            
            recs_found += 1
            if recs_found >= top_k:
                break
                
    return pd.DataFrame(recommendations)
